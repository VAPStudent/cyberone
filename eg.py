import ollama

# Query the Ollama model
response = ollama.chat(
    model="llama2",  # Specify the model (you can change it to other available models)
    messages=[{"role": "user", "content": "Hello, Ollama!"}]
)

# Print the response from the model
print(response)
