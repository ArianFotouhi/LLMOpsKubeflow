import kfp.dsl as dsl
from typing import List
from app.finetuneLLM import LLM
from app.ETL import ExTrLo

# Component 1: Extract and transform data
@dsl.component(
    base_image='python:3.8',
    packages_to_install=['requests', 'pandas', 'pyarrow']
)
def extract_and_transform() -> dsl.OutputPath(str):
    etl = ExTrLo()
    etl.extract()
    etl.transform()
    etl.load('/mnt/data/dataset.parquet')
    return '/mnt/data/dataset.parquet'


# Component 2: Fine-tune the model
@dsl.component(
    base_image='python:3.8',
    packages_to_install=['torch', 'transformers', 'peft', 'trl', 'datasets']
)
def fine_tune_model(dataset_file: dsl.InputPath(str)) -> None:
    llm = LLM(dataset_file=dataset_file)
    llm.setUpTrainer()
    llm.startFineTuner()
    prompt = 'What is the closing price of bitcoin today?'
    llm.evaluate(input_prompt=prompt)


# Define the pipeline
@dsl.pipeline(
    name='LLM Fine-tuning Pipeline',
    description='A pipeline for fine-tuning a LLM with music chart data'
)
def llm_finetuning_pipeline():
    # Step 1: Extract and transform data
    dataset = extract_and_transform()
    
    # Step 2: Fine-tune the model
    fine_tune_model(dataset_file=dataset.output)


# Compile the pipeline
if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(llm_finetuning_pipeline, 'llm_finetuning_pipeline.yaml')
