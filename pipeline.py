# pipeline.py

from src.automl_pipeline import AutoMLPipeline

if __name__ == "__main__":
    print("Menjalankan AutoML Pipeline...\n")
    automl = AutoMLPipeline(scoring="accuracy", cv=5)
    best_model, results = automl.run()

    print("\nHasil Evaluasi Semua Model:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics['mean_score']:.4f} ± {metrics['std']:.4f}")
    
    # Simpan model & hasil
    automl.save_model("models/best_model.pkl")
    automl.save_results("results/evaluation_report.json")
    
    print("Pipeline selesai!")