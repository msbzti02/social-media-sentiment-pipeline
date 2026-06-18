<div align="center">

# 📊 Social Media Sentiment Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-FF6F00.svg)](https://www.tensorflow.org/)
[![PySpark](https://img.shields.io/badge/PySpark-3.5+-E25A1C.svg)](https://spark.apache.org/)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-2.0+-231F20.svg)](https://kafka.apache.org/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.5+-47A248.svg)](https://www.mongodb.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Big Data and Artificial Intelligence pipeline designed to ingest, process, and classify sentiment from social media streams (Twitter and Reddit) using advanced Machine Learning and Deep Learning architectures.

**Author:** Mohammed Badawi

</div>

---

## 📖 Project Overview

The **Social Media Sentiment Analysis Pipeline** is a comprehensive, scalable system built to handle large volumes of textual data from various social platforms. The project covers the entire data lifecycle: from simulating real-time data ingestion using **Apache Kafka**, storing documents in **MongoDB**, processing large datasets with **Apache Spark**, to training state-of-the-art Natural Language Processing models including **TF-IDF + Logistic Regression**, **Bidirectional LSTMs (BiLSTM)**, and **HuggingFace BERT**. 

The final output is presented through an interactive **Streamlit Dashboard** that allows users to monitor real-time sentiment distribution and compare model performance metrics.

---

## ✨ Key Features

- **🚀 Automated Orchestration:** A centralized `run_pipeline.py` script to seamlessly trigger data loading, EDA, and preprocessing stages.
- **📡 Real-Time Streaming Simulation:** Kafka producers and consumers to handle high-throughput social media message queues.
- **🍃 Robust NoSQL Storage:** MongoDB integration for persisting raw streams and preprocessed features with optimized indexing.
- **⚡ Distributed Big Data Processing:** Apache Spark (PySpark) implementation for rapid exploratory data analysis and distributed MLlib pipelines.
- **🧠 Advanced Deep Learning Models:** Compares traditional ML (Logistic Regression) against deep sequence models (BiLSTM) and transformer-based architectures (BERT).
- **📊 Comprehensive EDA & Evaluation:** In-depth visualizations including word clouds, temporal trend analysis, ROC curves, and multi-model radar charts.
- **📈 Interactive UI:** A beautifully styled Streamlit dashboard providing real-time metric tracking and dynamic filtering.

---

## 🏗 Architecture Overview

1. **Data Ingestion:** Synthetic Twitter and Reddit data are generated/loaded and pushed into **Apache Kafka** topics.
2. **Persistence:** Kafka consumers push the raw data streams into **MongoDB**.
3. **Preprocessing:** Text is cleaned, tokenized, lemmatized, and enriched with VADER sentiment scores and metadata.
4. **Big Data Analysis:** **PySpark** connects to the data to perform distributed SQL queries and baseline MLlib training.
5. **Model Training:** Data is split to train Logistic Regression, TensorFlow BiLSTM, and PyTorch/HuggingFace BERT models.
6. **Evaluation & Visualization:** The models are evaluated using macro F1, ROC-AUC, and Cohen's Kappa, and the results are served via a **Streamlit** dashboard.

---

## 🛠 Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | Python 3 |
| **Data Engineering** | Apache Kafka, Apache Spark (PySpark), Pandas, Numpy |
| **Database** | MongoDB (PyMongo) |
| **Machine Learning** | Scikit-Learn |
| **Deep Learning** | TensorFlow (Keras), PyTorch, HuggingFace Transformers |
| **NLP** | NLTK, spaCy, vaderSentiment |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Frontend/UI** | Streamlit |

---

## 📂 Project Structure

```text
mohammed_SP_1/
├── config.py                   # Global configuration, paths, and hyperparameters
├── requirements.txt            # Project dependencies
├── run_pipeline.py             # Main orchestrator for initial pipeline stages
├── data/                       # Data loading and synthetic generation scripts
│   ├── load_data.py            
│   └── generate_twitter.py     
├── eda/                        # Exploratory Data Analysis and visualization
│   └── eda_analysis.py         
├── preprocessing/              # Text cleaning, tokenization, and TF-IDF extraction
│   └── preprocess.py           
├── mongodb/                    # MongoDB connection and aggregation queries
│   └── mongo_ops.py            
├── kafka/                      # Kafka producers, consumers, and in-memory simulation
│   ├── producer.py             
│   ├── consumer.py             
│   └── kafka_simulation.py     
├── spark/                      # PySpark distributed SQL and MLlib pipelines
│   └── spark_pipeline.py       
├── models/                     # ML/DL Model training and comprehensive evaluation
│   ├── logistic_regression.py  
│   ├── bilstm_model.py         
│   ├── bert_model.py           
│   └── evaluate.py             
└── dashboard/                  # Interactive Streamlit web application
    └── app.py                  
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- [MongoDB Community Server](https://www.mongodb.com/try/download/community) running on default port `27017`
- [Apache Kafka](https://kafka.apache.org/downloads) running on default port `9092` (Optional, script includes a simulation fallback)
- [Apache Spark](https://spark.apache.org/downloads.html) configured in your environment

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/social-media-sentiment.git
cd social-media-sentiment
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage Guide

The pipeline is divided into sequential stages. You can run the orchestrator for the initial setup, followed by individual component execution.

### Phase 1: Data Preparation & Preprocessing
Run the main pipeline to generate data, perform EDA, and preprocess the text.
```bash
python run_pipeline.py
```

### Phase 2: Database & Streaming
Start MongoDB integration and Kafka streams.
```bash
# Ensure MongoDB is running locally
python mongodb/mongo_ops.py

# Simulate Kafka streams (if a real broker isn't running)
python kafka/kafka_simulation.py
```

### Phase 3: Big Data & Machine Learning
Execute PySpark pipelines and train the sophisticated ML/DL models.
```bash
python spark/spark_pipeline.py
python models/logistic_regression.py
python models/bilstm_model.py
python models/bert_model.py
python models/evaluate.py
```

### Phase 4: Launch the Dashboard
Start the Streamlit UI to explore the results interactively.
```bash
streamlit run dashboard/app.py
```

---

## 🌍 Environment Variables

Create a `.env` file in the root directory (optional, currently configured via `config.py`):

```env
MONGO_URI=mongodb://localhost:27017/
MONGO_DB=sentiment_db
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
SPARK_MASTER=local[*]
```

*(Note: The project currently uses `config.py` for variables, but migrating sensitive endpoints to `.env` is recommended for production).*

---

## 📸 Screenshots

*(Placeholders for project screenshots)*

| Dashboard Overview | Model Evaluation Radar |
| :---: | :---: |
| <img src="https://via.placeholder.com/400x250?text=Streamlit+Dashboard" alt="Dashboard"/> | <img src="https://via.placeholder.com/400x250?text=Model+Comparison" alt="Evaluation"/> |

---

## 🔧 Troubleshooting

- **MongoDB Connection Refused:** Ensure your local MongoDB service is actively running (`mongod --dbpath /data/db`).
- **Kafka Not Found:** If you don't want to configure a full Kafka broker, run `python kafka/kafka_simulation.py` which utilizes a queue-based threaded simulation that mimics Kafka behavior perfectly.
- **NLTK Download Errors:** If the preprocessing stage fails due to missing NLTK datasets, open a python shell and run: `import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')`.
- **PySpark Memory Issues:** Adjust the `spark.driver.memory` configuration in `spark/spark_pipeline.py` or `config.py` if encountering Java heap space errors.

---

## 🔮 Future Improvements

- [ ] **Dockerization:** Wrap the entire pipeline (Kafka, Zookeeper, MongoDB, Streamlit) into a cohesive `docker-compose.yml`.
- [ ] **Cloud Deployment:** Migrate from local Spark to AWS EMR / Databricks and MongoDB to Atlas.
- [ ] **Live Twitter API Integration:** Replace synthetic Twitter generation with the actual X/Twitter Developer API (v2).
- [ ] **LLM Integration:** Evaluate zero-shot classification using OpenAI's GPT-4 or Meta's Llama-3.

---

## 🤝 Contributors

- **Mohammed Badawi** - *Lead Data Engineer & ML Developer*

Contributions are welcome! Please open an issue or submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
