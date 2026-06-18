import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

def create_spark_session():
    from pyspark.sql import SparkSession
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

def spark_sql_eda(spark, csv_path):
    print("\n  --- Spark SQL EDA ---")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)
    df.createOrReplaceTempView("posts")

    print(f"\n  Total records: {df.count():,}")
    print(f"  Schema:")
    df.printSchema()

    print("\n  Sentiment Distribution (Spark SQL):")
    spark.sql("""
        SELECT sentiment, COUNT(*) as count,
               ROUND(COUNT(*)*100.0 / (SELECT COUNT(*) FROM posts), 1) as pct
        FROM posts GROUP BY sentiment ORDER BY count DESC
    """).show()

    print("  By Source:")
    spark.sql("""
        SELECT source, sentiment, COUNT(*) as count
        FROM posts GROUP BY source, sentiment ORDER BY source, count DESC
    """).show()

    print("  Avg Word Count by Sentiment:")
    spark.sql("""
        SELECT sentiment,
               ROUND(AVG(SIZE(SPLIT(text, ' '))), 1) as avg_words,
               MAX(SIZE(SPLIT(text, ' '))) as max_words
        FROM posts GROUP BY sentiment
    """).show()

    return df

def spark_mllib_pipeline(spark, csv_path):
    from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator

    print("\n  --- Spark MLlib Pipeline ---")
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    from pyspark.sql.functions import when, col
    df = df.withColumn("label",
        when(col("sentiment")=="positive", 2.0)
        .when(col("sentiment")=="negative", 0.0)
        .otherwise(1.0))
    df = df.filter(col("text").isNotNull())

    train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)
    print(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")

    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    hashing_tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=10000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    lr = LogisticRegression(maxIter=20, regParam=0.01)
    pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, lr])

    print("  Training Logistic Regression (Spark MLlib)...")
    start = time.time()
    model = pipeline.fit(train_df)
    train_time = time.time() - start
    print(f"  Training time: {train_time:.1f}s")

    predictions = model.transform(test_df)
    evaluator_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(metricName="f1")
    accuracy = evaluator_acc.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    print(f"\n  Results:")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"    Training Time: {train_time:.1f}s")

    model_path = os.path.join(MODELS_DIR, "spark_lr_model")
    try:
        model.write().overwrite().save(model_path)
        print(f"    Model saved: {model_path}")
    except Exception as e:
        print(f"    [WARN] Model save skipped (Hadoop native): {type(e).__name__}")
        print(f"    Model training and evaluation completed successfully.")
    return model

if __name__ == "__main__":
    print("\n" + "="*60 + "\n  STAGE 7: APACHE SPARK PROCESSING\n  Mohammed Badawi - 2105656\n" + "="*60)

    csv_path = os.path.join(RAW_DATA_DIR, "combined_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"  [ERROR] Data not found at {csv_path}"); sys.exit(1)

    spark = create_spark_session()
    spark_sql_eda(spark, csv_path)
    spark_mllib_pipeline(spark, csv_path)
    spark.stop()
    print("\n" + "="*60 + "\n  STAGE 7 COMPLETE\n" + "="*60 + "\n")
