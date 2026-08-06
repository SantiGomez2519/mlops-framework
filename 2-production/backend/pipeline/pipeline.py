from backend.pipeline import FeatureEngineering, Preprocessing, Training


class Pipeline:
    def run(self) -> None:
        print("Step 1/3: preprocessing")
        Preprocessing().run()

        print("Step 2/3: feature engineering")
        featured, scaler = FeatureEngineering().run()

        print("Step 3/3: training")
        Training().run(featured, scaler)
