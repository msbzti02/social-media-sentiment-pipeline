import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *

def run_stage(stage_num, name, module_path):
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: {name}")
    print(f"{'='*60}")
    start = time.time()
    try:
        exec(open(module_path, encoding="utf-8").read(), {"__name__": "__main__"})
        elapsed = time.time() - start
        print(f"\n  ✓ Stage {stage_num} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ✗ Stage {stage_num} FAILED after {elapsed:.1f}s: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SOCIAL MEDIA SENTIMENT ANALYSIS PIPELINE")
    print("  Mohammed Badawi - 2105656")
    print("  Big Data & Artificial Intelligence")
    print("="*60)

    stages = [
        (2, "Dataset Loading", os.path.join(PROJECT_ROOT, "data", "load_data.py")),
        (3, "Exploratory Data Analysis", os.path.join(PROJECT_ROOT, "eda", "eda_analysis.py")),
        (4, "Data Preprocessing", os.path.join(PROJECT_ROOT, "preprocessing", "preprocess.py")),
    ]

    results = {}
    total_start = time.time()

    for stage_num, name, path in stages:
        if os.path.exists(path):
            success = run_stage(stage_num, name, path)
            results[stage_num] = "✓ PASS" if success else "✗ FAIL"
        else:
            print(f"\n  [SKIP] Stage {stage_num}: {path} not found")
            results[stage_num] = "⊘ SKIP"

    total_time = time.time() - total_start

    print("\n\n" + "="*60)
    print("  PIPELINE SUMMARY")
    print("="*60)
    for stage, status in results.items():
        print(f"  Stage {stage}: {status}")
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n  NEXT STEPS:")
    print("  - Stage 5: Start MongoDB → python mongodb/mongo_ops.py")
    print("  - Stage 6: Start Kafka  → python kafka/producer.py")
    print("  - Stage 7: Run Spark    → python spark/spark_pipeline.py")
    print("  - Stage 8: Train models → python models/logistic_regression.py")
    print("  -                       → python models/bilstm_model.py")
    print("  -                       → python models/bert_model.py")
    print("  - Stage 9: Dashboard    → streamlit run dashboard/app.py")
    print("="*60 + "\n")
