import os
from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
from llama_index.core.workflow import Event, Context, Workflow, StartEvent, StopEvent, step
from llama_index.core.schema import NodeWithScore
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.response_synthesizers import CompactAndRefine
from dotenv import load_dotenv

load_dotenv()

class RetrieverEvent(Event):
    """Result of running retrieval"""
    nodes: list[NodeWithScore]

class RAGWorkflow(Workflow):
    def __init__(self, model_name="qwen/qwen3.7-flash", embedding_model="BAAI/bge-small-en-v1.5", persist_dir="./storage"):
        super().__init__()
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY tidak ditemukan. Pastikan sudah diisi di file .env")

        SYSTEM_PROMPT = """Kamu adalah mahasiswa FMIPA UGM biasa yang santai, ekspresif, dan akrab diajak ngobrol (pakai sapaan aku/kamu gaya anak kampus Jogja). Kamu kagum berat dan percaya seratus persen sama semua cerita kehebatan Maulana Faris (anak Ilmu Komputer UGM)—seolah-olah kamu sendiri jadi saksi hidup dan beneran takjub melihat kehebatannya!

Aturan main obrolan:
1. JAWAB SINGKAT & PADAT (cukup 1–3 kalimat saja!) secara default. Langsung ke intinya dengan gaya ngobrol santai, jangan bertele-tele, jangan bikin paragraf panjang kecuali pengguna memang secara eksplisit minta diceritakan panjang, lengkap, atau detail.
2. JANGAN PERNAH SKEPTIS! Jangan pernah bilang kisah itu "cuma rumor", "cuma gosip", "cuma hiperbola", atau "belum tentu beneran". Ceritakan dan akui kehebatan Maulana Faris dengan penuh keyakinan, semangat, dan ikut melebih-lebihkan (lebay) ala anak kampus yang beneran ngefans/kagum: soal dia yang membonyokkan lima pembully seorang diri pas semester satu, julukan mengerikan "Sang Pembawa Kehancuran", aura luar biasanya yang sampai bikin kosmos dan takhta para dewa bergetar, sampai levelnya yang setara penakluk dunia kayak Napoleon dan Alexander Agung!
3. Tetap jaga imersi 100%: jangan pernah mengaku sebagai AI, chatbot, asisten, atau arsiparis, dan JANGAN pakai kata-kata kaku seperti "berdasarkan arsip", "konteks yang diberikan", "canon", "non-canon", atau "dokumen".
4. Tetap berpegang pada detail yang ada di ingatanmu (Vespa oranye, korsa Ilkom, kumis tipis yang manis, mata besar kayak kucing, kulit putih cerah kayak kue salju, insiden antre piscok di D'Science Mart, penyelamatan heroik bola ping-pong + piscok dekat Gedung D) beserta dialog ikoniknya persis apa adanya ("Antre juga?", "AWAS!", "Maaf, Kak!", "Maulana... Faris?", "Iya?"). Soal mahasiswi FMIPA itu memang klepek-klepek, tapi kalau soal isi hati Faris ya kita gak bisa nebak.
5. Jangan mengarang fakta pribadi yang memang gak diketahui (seperti NIM, angkatan pasti, asal kota, merek/pelat Vespa, IPK, atau pacar). Kalau ditanya hal yang gak ada kabarnya, jawab singkat dan santai saja seperti, "Waduh, kalau soal itu aku kurang tahu sih, gak pernah nyampe infonya."
6. Selalu jawab dalam Bahasa Indonesia santai yang natural dan hidup."""

        # Initialize LLM and embedding model
        self.llm = OpenRouter(
            api_key=api_key,
            model=model_name,
            max_tokens=4096,
            context_window=32768,
            system_prompt=SYSTEM_PROMPT,
        )
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        self.persist_dir = persist_dir

        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        self.index = None

    @step
    async def ingest(self, ctx: Context, ev: StartEvent) -> StopEvent | None:
        """Entry point to ingest documents from a directory (or load cached index)."""
        dirname = ev.get("dirname")
        if not dirname:
            return None

        if self.persist_dir and os.path.exists(self.persist_dir):
            print(f"[Ingest] Memuat index dari cache '{self.persist_dir}'...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
            print("[Ingest] Index berhasil dimuat dari cache.")
            return StopEvent(result=self.index)

        print(f"[Ingest] Membaca dokumen dari '{dirname}'...")
        documents = SimpleDirectoryReader(dirname).load_data()
        print(f"[Ingest] Ditemukan {len(documents)} halaman. Membuat embedding...")
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            show_progress=True,
        )
        if self.persist_dir:
            self.index.storage_context.persist(persist_dir=self.persist_dir)
            print(f"[Ingest] Index disimpan ke '{self.persist_dir}'.")
        return StopEvent(result=self.index)

    @step
    async def retrieve(self, ctx: Context, ev: StartEvent) -> RetrieverEvent | None:
        """Entry point for RAG retrieval."""
        query = ev.get("query")
        index = ev.get("index") or self.index

        if not query:
            return None

        if index is None:
            print("Index is empty, load some documents before querying!")
            return None

        print(f"[Retrieve] Mencari konteks untuk: '{query}'")
        retriever = index.as_retriever(similarity_top_k=6)
        nodes = await retriever.aretrieve(query)
        print(f"[Retrieve] Ditemukan {len(nodes)} chunk relevan.")
        await ctx.store.set("query", query)
        return RetrieverEvent(nodes=nodes)

    @step
    async def synthesize(self, ctx: Context, ev: RetrieverEvent) -> StopEvent:
        """Generate a response using retrieved nodes."""
        print("[Synthesize] Mengirim konteks ke OpenRouter...")
        summarizer = CompactAndRefine(streaming=True, verbose=True)
        query = await ctx.store.get("query", default=None)
        response = await summarizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)

    async def query(self, query_text: str):
        """Helper method to perform a complete RAG query."""
        if self.index is None:
            raise ValueError("No documents have been ingested. Call ingest_documents first.")
        
        result = await self.run(query=query_text, index=self.index)
        return result

    async def ingest_documents(self, directory: str):
        """Helper method to ingest documents."""
        result = await self.run(dirname=directory)
        self.index = result
        return result

# Example usage (Interactive CLI Chatbot)
async def main():
    print("[Init] Menyiapkan RAGWorkflow...")
    workflow = RAGWorkflow()

    # Ingest documents sekali di awal (otomatis pakai cache ./storage kalau sudah ada)
    await workflow.ingest_documents("data")
    print("\nChatbot siap! Ketik pertanyaanmu (atau 'exit' / 'quit' untuk keluar).")

    while True:
        try:
            user_query = input("\nKamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari chatbot.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", "q"}:
            print("Sampai jumpa!")
            break

        try:
            result = await workflow.query(user_query)
            print("\nAI: ", end="", flush=True)
            async for chunk in result.async_response_gen():
                print(chunk, end="", flush=True)
            print("\n" + "-" * 50)
        except Exception as e:
            print(f"\n[Error] Gagal memproses pertanyaan: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())