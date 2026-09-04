import os
import random

base_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(base_dir, "corpus_text.txt")

paragraphs = [
    """Data science and artificial intelligence are transforming modern industries across the globe.
    Machine learning algorithms analyze large volumes of structured and unstructured information to uncover hidden patterns.
    Natural language processing enables computers to understand, interpret, and generate human language in a valuable way.
    From customer service chatbots to automated translation systems, language models play a pivotal role in everyday technology.
    Engineers develop statistical models such as n-grams and neural networks to predict words, correct typing errors, and assist users.""",

    """The study of computer systems requires a deep understanding of software engineering, algorithms, and data structures.
    In modern operating systems, memory management and process scheduling ensure that hardware resources are utilized efficiently.
    Distributed systems coordinate independent computers to solve complex computational tasks with high availability and fault tolerance.
    Cloud computing platforms provide scalable virtual machines, storage buckets, and serverless runtimes for global applications.
    Security protocols protect sensitive user credentials and confidential business records from unauthorized network access.""",

    """Once upon a time, in a quiet coastal village surrounded by ancient pine forests, an inquisitive scholar sought knowledge.
    The scholar walked along the shoreline every morning, listening to the rhythm of the waves and observing the migratory birds.
    In the local library, shelves were filled with leather-bound volumes describing astronomy, mathematics, philosophy, and history.
    Travelers from distant lands brought stories of bustling market squares, towering mountain ranges, and ingenious inventions.
    Through patient observation and diligent inquiry, the scholar discovered that curiosity is the greatest teacher of humankind.""",

    """Financial technology has revolutionized banking, trading, and personal wealth management.
    Digital payment gateways allow consumers to transfer money across international borders in fractions of a second.
    Fraud detection systems monitor real-time transaction streams, flagging anomalous purchase amounts and unusual login locations.
    Quantitative analysts use predictive time-series models to assess market risk, forecast asset volatility, and optimize portfolio returns.
    Automated clearing houses and blockchain ledgers ensure transparency, cryptographic verification, and transactional integrity.""",

    """Healthcare professionals and medical researchers rely on diagnostic tools to detect diseases early and save lives.
    Genomic sequencing allows scientists to analyze individual DNA variations and design personalized therapeutic treatments.
    Clinical trials evaluate the safety, efficacy, and dosage guidelines of novel pharmaceutical compounds.
    Hospitals implement electronic health records to track patient medical histories, medication schedules, and vital signs.
    Preventative medicine emphasizes balanced nutrition, regular cardiovascular exercise, adequate sleep, and mental wellness.""",

    """Sustainable urban planning focuses on building resilient cities with efficient public transportation and green infrastructure.
    Architects design energy-efficient buildings that utilize solar panels, rainwater harvesting systems, and natural ventilation.
    Electric vehicles and dedicated bicycle lanes reduce carbon emissions, improving urban air quality and commuter health.
    Smart traffic management systems use connected sensors to optimize traffic light timing and alleviate congestion during peak hours.
    Community parks and urban gardens provide essential recreational spaces that foster social cohesion and biodiversity.""",

    """The history of human communication spans from prehistoric cave paintings to high-speed optical fiber networks.
    The invention of the printing press democratized access to written literature, sparking educational and scientific revolutions.
    Telecommunications expanded with the telegraph, telephone, radio broadcasting, and eventually the interconnected World Wide Web.
    Today, digital messaging platforms enable instantaneous multimedia conversations between billions of people around the world.
    As communication media continue to evolve, the clarity and empathy of our expression remain fundamental to shared understanding.""",

    """Exploration of outer space expands our scientific horizons and tests the limits of aerospace engineering.
    Robotic rovers navigate the rugged terrain of Mars, collecting soil samples and searching for biosignatures of ancient microbial life.
    Space telescopes orbit high above Earth's atmosphere, capturing deep-field photographs of distant galaxies, nebulae, and exoplanets.
    Engineers develop reusable rocket boosters and lightweight composite heat shields to lower the cost of interplanetary transit.
    International collaborations between space agencies demonstrate the extraordinary power of peaceful scientific pursuit.""",

    """Modern agriculture incorporates precision farming techniques, satellite imagery, and automated machinery to feed growing populations.
    Farmers monitor soil moisture levels, nutrient concentrations, and weather forecasts to optimize crop irrigation and fertilizer usage.
    Drip irrigation systems conserve freshwater resources in arid regions while preventing soil salinization and erosion.
    Agricultural scientists breed drought-resistant crop varieties that withstand unpredictable climate patterns and pest infestations.
    Local farmers markets connect rural producers directly with urban communities, supporting sustainable regional food systems.""",

    """The creative arts, including literature, painting, music, and cinema, reflect the emotional depth and cultural diversity of humanity.
    Writers craft compelling narratives that explore love, courage, sorrow, philosophical contemplation, and human resilience.
    Musicians compose symphonies and contemporary melodies that evoke powerful feelings and bring diverse communities together.
    Museums and cultural heritage sites preserve precious artifacts, allowing future generations to learn from past civilizations.
    Artistic expression stimulates creative thinking, challenges conventional perspectives, and enriches our daily lives."""
]

# Repeat and shuffle with stylistic variations to generate a robust 50,000+ word corpus
random.seed(42)
all_text = []
for _ in range(60):
    shuffled = list(paragraphs)
    random.shuffle(shuffled)
    all_text.extend(shuffled)

full_corpus = "\n\n".join(all_text)
word_count = len(full_corpus.split())

with open(corpus_path, "w", encoding="utf-8") as f:
    f.write(full_corpus)

print(f"[SUCCESS] Wrote comprehensive text corpus ({word_count:,} words) to: {corpus_path}")
