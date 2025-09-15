# ChatVis

ChatVis has two components: ChatVis_agent and ChatVis_benchmark.

- ChatVis_agent is an LLM agent that aids the LLM to generate Python code for ParaView scientific visualization tasks. The agent currently uses GPT-4o as the underlying LLM, but one can easily modify the agent code to call a different LLM.
- ChatVis_benchmark is a benchmark suite of canonical visualization tasks, ParaView regression tests, and scientific use cases that includes comprehensive evaluation metrics. The benchmark can be used independently of ChatVis_agent to evaluate other agents and LLMs.

The `ChatVis_agent` and `ChatVis_benchmark` directories each contain a README with more details.
