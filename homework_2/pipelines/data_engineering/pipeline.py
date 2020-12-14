from kedro.pipeline import Pipeline, node

from homework_2.pipelines.data_engineering.nodes import load_data


def create_pipeline():
    return Pipeline(
        [
            node(
                func=load_data,
                inputs=["titanic_train_clean.csv", "params:example_test_data_ratio"],
                outputs=dict(
                    train_x="example_train_x",
                    train_y="example_train_y",
                    test_x="example_test_x",
                    test_y="example_test_y",
                ),
            ),
        ]
    )


if __name__ == "__main__":
    print(create_pipeline())
