import os
import google.generativeai as genai
from google.generativeai import types

project = os.getenv("GOOGLE_CLOUD_PROJECT", "potent-trail-454820-h5")
location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

def query_flash(question = "What is APR?", context_chunks = [], model_name="gemini-2.0-flash-001", top_k=3):
    try:
        # Input validation
        if not isinstance(question, str) or not question.strip():
            raise ValueError("The question must be a non-empty string.")
        if not isinstance(context_chunks, list) or not all(
            isinstance(chunk, (list, tuple)) for chunk in context_chunks
        ):
            raise ValueError("Context chunks must be a list of tuples or lists.")

        # Construct the context
        context = " ".join([context_chunk[0] for context_chunk in context_chunks])

        # Prompt Engineering
        prompt = (
            f"You are a legal assistant specializing in contracts. "
            f"Answer the question based on the following context, and cite the sources explicitly. "
            f"Do not include any information not present in the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

        model = genai.GenerativeModel(model_name)
        contents = [prompt]
        generation_config = genai.GenerationConfig(
            temperature = 1,
            top_p = 0.95,
            max_output_tokens = 8192,

        )
                
            

        # Generate content (streaming)
        response_stream = model.generate_content(contents, generation_config=generation_config)

        # Collect the streamed response
        generated_answer = ""
        for chunk in response_stream:
            generated_answer += chunk.text

        relevant_chunks = context_chunks[:top_k]  # Top k chunks for citation
        return {"answer": generated_answer.strip(), "relevant_context": relevant_chunks}

    except genai.types.generation_types.BlockedPromptException as e:
        return f"Blocked Prompt Error: {e}"
    except Exception as e:
        return {"answer": f"An unexpected error occurred: {e}", "relevant_context": []}


if __name__ == "__main__":
    # Example usage (replace with your actual data)
    question = "What is the main topic of this document?"
    context_chunks = [("This document discusses the importance of AI in healthcare.", 0.8, 1),
                      ("AI can improve diagnostic accuracy and treatment outcomes.", 0.7, 2),
                      ("Ethical considerations are crucial when implementing AI in healthcare.", 0.6, 3)]

    result = query_flash(question, context_chunks)
    print(result)

