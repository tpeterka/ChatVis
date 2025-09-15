# ChatVis_agent

ChatVis_agent is an LLM agent that aids the LLM to generate Python code for ParaView scientific visualization tasks.
The agent currently uses GPT-4o as the underlying LLM, but one can easily modify the agent code to call a different LLM.
ChatVis does not require retraining or fine-tuning the LLM.
Instead, ChatVis employs chain-of-thought prompt simplification, retrieval-augmented prompt generation using a vector database of documentation and code examples, and error checking with iterative prompt feedback to correct errors until a visualization is produced.

More details and instructions to come.
