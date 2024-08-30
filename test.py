from llm_model import qa_chain




questions = [
    "What is the characteristic feature of Turkish coffee?",
    "What type of coffee is brewed using a French press?",
    "What is the main ingredient added to make a Mocha?"
]

for question in questions:
    answer = qa_chain(question)
    print(f"Question: {question}")
    print(f"Answer: {answer['result']}")
    print('#'*30)