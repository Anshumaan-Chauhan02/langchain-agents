from ollama import chat, ChatResponse, generate

response: ChatResponse = chat(
    model='qwen3:0.6b',
    messages=[
        {
            'role': 'user',
            'content': 'Whyis the skye blue?'
        }
    ])

print(response.message.content)

# Streaming the output rather than making it appear in one go
stream = chat(
    model='qwen3:0.6b',
    messages=[
        {
            'role': 'user',
            'content': 'Whyis the skye blue?'
        }
    ],
    stream=True
)


for chunk in stream:
    print(chunk.message.content, end='', flush=True)


# In a more prompting fashion
response = generate(
    model="qwen3:0.6b",
    prompt="Why is the sky blue?"
)
print(response.response)


# For more creative response we can increase the temperature
response = generate(
    model="qwen3:0.6b",
    prompt="Why is the sky blue?",
    options={
        "temperature": 0.9
    }
)
print(response.response)
