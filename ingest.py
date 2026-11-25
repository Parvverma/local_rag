
import pymupdf
import ollama
import chromadb

from pathlib import Path
import tempfile


# ==========================================
# CONFIGURATION
# ==========================================

PDF_DIR = Path("documents")
DB_DIR = "./chroma_db"

LLM_MODEL = "gemma4:e4b"
EMBEDDING_MODEL = "embeddinggemma"

COLLECTION_NAME = "pdf_knowledge"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 50

# Enable visual understanding of every PDF page
ENABLE_VISION = True

# OCR language
OCR_LANGUAGE = "eng"


# ==========================================
# INITIALIZE CHROMADB
# ==========================================

client = chromadb.PersistentClient(
    path=DB_DIR
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ==========================================
# STEP 1: EXTRACT TEXT + OCR
# ==========================================

def extract_page_text(page):

    # First, extract normal digital text
    normal_text = page.get_text("text").strip()

    # Check whether page contains embedded images
    has_images = bool(page.get_images(full=True))

    # Run OCR when:
    # 1. Page contains little/no digital text
    # 2. Page contains images that may have text

    if len(normal_text) < 200 or has_images:

        print(f"Running OCR on page {page.number + 1}")

        try:

            ocr_page = page.get_textpage_ocr(
                language=OCR_LANGUAGE,
                dpi=250,
                full=not bool(normal_text)
            )

            ocr_text = page.get_text(
                textpage=ocr_page
            ).strip()

            if ocr_text:
                return ocr_text

        except Exception as error:

            raise RuntimeError(
                "OCR failed. Check that Tesseract is installed "
                f"and available in PATH. Original error: {error}"
            ) from error

    return normal_text


# ==========================================
# STEP 2: UNDERSTAND IMAGES AND DIAGRAMS
# ==========================================

def analyze_page_with_gemma(page, output_dir):

    # Render complete PDF page as an image
    pixmap = page.get_pixmap(
        matrix=pymupdf.Matrix(1.5, 1.5),
        alpha=False
    )

    image_path = output_dir / f"page_{page.number + 1}.png"

    pixmap.save(str(image_path))

    print(
        f"Analyzing visual content on page {page.number + 1}"
    )

    prompt = """
    Analyze this PDF page carefully.

    Focus on visual information:

    1. Explain diagrams and their components.
    2. Explain flowcharts and the sequence of steps.
    3. Describe graphs, axes, legends and trends.
    4. Describe tables and their relationships.
    5. Explain technical illustrations.
    6. Mention important labels and symbols.

    Do not invent information that is not visible.

    Avoid copying ordinary paragraphs of text.
    Focus on information that cannot be understood
    from normal text extraction alone.

    Write a detailed, self-contained description.
    """

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(image_path)]
            }
        ]
    )

    return response.message.content.strip()


# ==========================================
# STEP 3: CREATE CHUNKS
# ==========================================

def create_chunks(text, metadata):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk_text = text[start:end].strip()

        if chunk_text:

            chunks.append({
                "text": chunk_text,
                **metadata
            })

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ==========================================
# STEP 4: PROCESS A SINGLE PDF
# ==========================================

def process_pdf(pdf_path):

    print("\n" + "=" * 50)
    print(f"Processing: {pdf_path.name}")
    print("=" * 50)

    doc = pymupdf.open(pdf_path)

    all_chunks = []

    with tempfile.TemporaryDirectory() as temp:

        output_dir = Path(temp)

        for page_index, page in enumerate(doc):

            page_number = page_index + 1

            print(f"\nProcessing page {page_number}")

            # --------------------------------
            # A. NORMAL TEXT + OCR
            # --------------------------------

            text = extract_page_text(page)

            if text:

                text_chunks = create_chunks(
                    text,
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "content_type": "text"
                    }
                )

                all_chunks.extend(text_chunks)

            # --------------------------------
            # B. VISUAL UNDERSTANDING
            # --------------------------------

            if ENABLE_VISION:

                try:

                    description = analyze_page_with_gemma(
                        page,
                        output_dir
                    )

                    if description:

                        visual_chunks = create_chunks(
                            description,
                            {
                                "source": pdf_path.name,
                                "page": page_number,
                                "content_type": "visual_description"
                            }
                        )

                        all_chunks.extend(visual_chunks)

                except Exception as error:

                    print(
                        f"WARNING: Vision failed on page "
                        f"{page_number}: {error}"
                    )

    doc.close()

    print(f"\nTotal chunks created: {len(all_chunks)}")

    return all_chunks


# ==========================================
# STEP 5: GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(texts):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts
    )

    return response["embeddings"]


# ==========================================
# STEP 6: STORE IN CHROMADB
# ==========================================

def store_chunks(chunks, pdf_name):

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        texts = [
            item["text"]
            for item in batch
        ]

        embeddings = generate_embeddings(texts)

        ids = [
            (
                f"{pdf_name}_"
                f"p{item['page']}_"
                f"{item['content_type']}_"
                f"c{start + index}"
            )
            for index, item in enumerate(batch)
        ]

        metadatas = [
            {
                "source": item["source"],
                "page": item["page"],
                "content_type": item["content_type"]
            }
            for item in batch
        ]

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"Stored {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)} chunks"
        )


# ==========================================
# MAIN FUNCTION
# ==========================================

def main():

    if not PDF_DIR.exists():

        print("Documents folder not found!")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:

        print("No PDF files found!")
        return

    print(f"Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:

        chunks = process_pdf(pdf_path)

        if chunks:

            store_chunks(
                chunks,
                pdf_path.stem
            )

    print("\n" + "=" * 50)
    print("ALL DOCUMENTS PROCESSED SUCCESSFULLY!")
    print("Total stored documents:", collection.count())
    print("=" * 50)


if __name__ == "__main__":
    main()
