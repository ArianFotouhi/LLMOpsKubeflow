# LLMops pipeline to fine tune llama3 on the daily API data

In this work, the pipeline begins by fetching and preparing the dataset using a selected API. Specifically, our example utilizes an API that provides the daily music chart and track metadata from Deezer. Next, the chosen language model (Llama3 in our case) is retrieved from Hugging Face, and the prepared dataset is used to fine-tune this model. Finally, the fine-tuned model is utilized for inferences.


All these steps are orchestrated within a Kubeflow pipeline. To deploy this pipeline, the generated YAML file should be uploaded to the Kubeflow UI. 

Also here are the requirements and packages to run the code:
```
!pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 kfp==2.7.0
```
