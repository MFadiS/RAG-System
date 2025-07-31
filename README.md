# Recipe RAG System with Ollama.ai

## Table of Contents
- [Introduction](#introduction)
- [Installing Ollama](#installing-ollama)
- [Domain and Dataset](#domain-and-dataset)
- [Data Ingestion](#data-ingestion)
- [Chunking and Embedding](#chunking-and-embedding)
- [Prompt Design](#prompt-design)
- [Setting up Retrieval](#setting-up-retrieval)
- [Testing](#testing)
- [Limitations](#limitations)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Retrieval Augmented Generation (RAG) is a powerful technique that enhances the accuracy and reliability of generative AI models by integrating factual information from external sources (Merritt, 2023). This approach allows models to leverage their natural language generation capabilities while responding to domain-specific queries.

This project focuses on training Ollama.ai's "mistral" Large Language Model (LLM) using the publicly available RecipeNLG dataset from Kaggle (Mooney, n.d.). Our primary goal is to develop a RAG framework capable of retrieving and generating recipes in response to user queries. This involves ingesting recipe data from a PDF file, processing and storing it for swift retrieval, and utilizing the LLM to generate informative responses. This RAG system aims to streamline recipe retrieval and generation, offering a promising solution to culinary exploration.

## Installing Ollama
To begin, you need to install and run Ollama on your local environment. You can download it from [https://ollama.com](https://ollama.com). Please note that while initial attempts were made on Google Colab, issues with Ollama server error codes (99 and 404) led to the transition of the code to Jupyter Notebook for execution on a local system.

## Domain and Dataset
The dataset central to this RAG system is RecipeNLG, sourced from Kaggle (Mooney, n.d.). This extensive dataset comprises 2,231,142 cooking recipes and was originally used for research in semi-structured text generation (Bień et al., 2020). Due to computational constraints, a randomly sampled subset of 10,000 recipes from the RecipeNLG dataset was used to train the LLM.

## Data Ingestion
The sampled dataset was converted into a PDF format using the R programming language. Subsequently, the `UnstructuredPDFLoader` package from LangChain (www.restack.io, n.d.) was employed within the Python environment to integrate the data into the RAG system. `UnstructuredPDFLoader` was chosen for its ease of use, adaptability within the LangChain framework, and compatibility with diverse file formats, simplifying PDF document loading and subsequent analysis.

## Chunking and Embedding
After data ingestion, the next step is chunking, which involves segmenting the document into smaller, more manageable sections that fit within the LLM's context window (Eteimorde, 2023). `RecursiveCharacterTextSplitter` from LangChain was selected as the optimal choice (Eteimorde, 2023). By fine-tuning hyperparameters like `chunk size` (7500 characters) and `chunk overlap` (100 characters), the process yielded refined chunks.

For embedding, Ollama.ai's `nomic-embed-text` model was chosen due to its superior performance across various embedding dimensions and efficient memory reduction, as highlighted by Rastogi (2024). Nomic Embed, based on BERT with additional optimizations, provides an efficient solution for text embedding.

The embedded representations are stored in a vector database created using `chromadb`. Chroma is an open-source database known for its ease of use, scalability, and efficiency in storing and retrieving vector embeddings (Dwyer, 2023), which is crucial for the seamless operation of a RAG system. The process resulted in 1098 chunks being embedded by the Nomic Embed model.

## Prompt Design
The prompt design is a critical component for effective user-AI interaction within the RAG system. The prompt template sets clear expectations for users and guides the AI model in generating relevant responses.

A `PromptTemplate` object is created with an input variable for the user's question. The template defines the AI assistant's role and task:
- "You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database."
- "By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search."

This design facilitates seamless interaction and knowledge discovery within the RAG system.

## Setting up Retrieval
This section sets up the retriever component of the RAG system. An instance of the `ChatOllama` class is created, specifying "mistral" as the LLM to be used for response generation. The `MultiQueryRetriever` is then instantiated using the `from_llm` method, configured with the vector database as the document retrieval source and the `query_prompt` template to guide the retrieval process.

Finally, a processing chain is constructed to control the flow of information and interactions within the RAG system. This chain includes:
- **Retriever:** For document retrieval.
- **Prompt:** For guiding the response generation process.
- **LLM:** For generating responses.
- **Output Parser:** For parsing the output into a readable format.

## Testing
To test the efficacy of the RAG system, five test queries were posed, including one deliberately outside the system's dataset. For the out-of-context query, the system adeptly recognized the lack of information and transparently indicated its limitation before drawing from its internal knowledge base. For the other queries, the system delivered precise and detailed responses consistent with the provided dataset, demonstrating its proficiency in harnessing available data for accurate recipe recommendations.

## Limitations
- **Scalability and Performance:** Downsizing the dataset was necessary due to substantial computational demands and cost challenges with the original 2 million+ recipes.
- **Contextual Understanding:** While proficient in retrieving relevant recipes, the system's capacity to understand nuanced or ambiguous queries may be constrained, potentially leading to irrelevant responses.
- **Response Generation Capability:** The system's ability to generate responses is limited by the training dataset. Without access to updated data, it cannot integrate new recipes or adapt to evolving culinary preferences.

## Conclusion
The Retrieval Augmented Generation (RAG) system developed in this project showcases the potential of integrating large language models with external data sources for domain-relevant responses. Leveraging the RecipeNLG dataset and Ollama.ai's mistral LLM, the system effectively provides precise recipe recommendations.

While promising, there are opportunities for further enhancement, including addressing scalability concerns and incorporating mechanisms for continuous learning and adaptation. This RAG system serves as a testament to the transformative potential of combining state-of-the-art language models with domain-specific knowledge bases, paving the way for more intuitive and knowledge-driven human-machine interactions.

## References
- Bien, M., Gilski, M., Maciejewska, M., Taisner, W., Wisniewski, D. and Lawrynowicz, A. (2020). *RecipeNLG: A Cooking Recipes Dataset for Semi-Structured Text Generation*. [online] ACLWeb. doi:https://doi.org/10.18653/v1/2020.inlg-1.4.
- Dwyer, J. (2023). *Exploring Chroma Vector Database Capabilities | Zeet.co*. [online] zeet.co. Available at: https://zeet.co/blog/exploring-chroma-vector-database-capabilities#:~:text=Chroma%20is%20an%20open%2Dsource [Accessed 2 Jun. 2024].
- Eteimorde, Y. (2023). *Understanding LangChain’s RecursiveCharacterTextSplitter*. [online] DEV Community. Available at: https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846.
- Merritt, R. (2023). *What Is Retrieval-Augmented Generation?* [online] NVIDIA Blog. Available at: https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/.
- Mooney, P. (n.d.). *RecipeNLG (cooking recipes dataset)*. [online] www.kaggle.com. Available at: https://www.kaggle.com/datasets/paultimothymooney/recipenlg/suggestions?status=pending&yourSuggestions=true [Accessed 2 Jun. 2024].
- Rastogi, R. (2024). *Papers Explained 110: Nomic Embed*. [online] Medium. Available at: https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#:~:text=dataset%20using%20MRL.- [Accessed 2 Jun. 2024].
- www.restack.io. (n.d.). *LangChain unstructured PDF loader*. [online] Available at: https://www.restack.io/docs/langchain-knowledge-langchain-unstructured-pdf-loader [Accessed 2 Jun. 2024].
