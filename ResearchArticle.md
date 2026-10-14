Deploying an agent is straightforward. Establishing whether it works, and whether it still works after a prompt revision or a model upgrade, is much harder.

This post is a theoretical introduction to that problem. It covers what an agentic eval is, how an eval harness is put together, which metrics matter in production, and the failure modes that leave a team with an eval system that reports nothing useful. Later posts in the series go into each of these in depth.

Everything here concerns production agentic systems serving end users: support agents, workflow agents, in-product assistants. Long-running coding agents operating inside a harness share many of the same principles but raise separate evaluation problems, and they are out of scope.

### **Defining an agentic eval**

Agentic evals are automated integration tests for AI agents. They verify how an agent behaves across the whole lifecycle of a user request, including the path it took to get to an answer rather than just the answer itself.

That is what separates them from LLM evals. An LLM eval tests a single inference: a prompt goes in, a response comes back, and the response is graded. An agent runs a loop of many inferences, calls tools, and changes external state along the way. Agentic evals therefore grade the whole trajectory.

An LLM call produces one output to grade. An agent produces a trajectory.

An LLM call produces one output to grade. An agent produces a trajectory.

An LLM call produces one output to grade. An agent produces a trajectory.

Errors also compound. If every step in a trajectory is independently 95% reliable, a ten-step task succeeds roughly 60% of the time, and an error early on corrupts everything downstream. Per-step accuracy that looks excellent in isolation can still give you an unreliable agent, which is why grading the final response is not enough signal on its own.

### **Evaluation criteria**

LLM evals usually assess three things:

- **Fulfillment.** Did the response satisfy the user’s request?
- **Groundedness.** Did the model hallucinate? Did every claim trace back to a source or a tool output?
- **Safety and compliance.** Did the response breach policy (like privacy law)?

Agentic evals add four more:

- **Trajectory.** Did the agent take a sensible sequence of steps to reach the outcome?
- **Tool use.** Did it pick the right tool, pass correct arguments, call tools in a valid order, and handle tool errors?
- **Side effects.** Did external state change as intended, and did anything change that should not have?
- **Efficiency.** Tokens per task, tool calls per task, agent turns to resolution, p50 and p95 latency, and cost per resolved task.

Custom metrics are rarely worth the effort at the start. Vertex evaluation service, the OpenAI Evals API and the LangSmith SDK all ship agent evaluation frameworks with predefined metrics. Start with those and tune them to your use case.

### **Categories of agent**

The kind of agent you are building sets the difficulty of evaluating it.

**Goal-oriented agents** are built around a bounded task: customer care, code review triage, returns processing. Side effects exist but stay simple, like sending an email or opening a ticket. Requests are mostly independent of one another and get served the same way as any other API call on a shared server.

**Long-running agents** take a complex task, run for minutes or hours, and return a result. Deep research agents and coding agents sit here. State is substantial, with files written, steps executed and memory accumulated, and side effects are both numerous and consequential. Concurrent tasks need real isolation, usually a VM or a container per run, or they interfere with each other.

Evals for long-running agents are much harder to build. You have stateful environments, memory that persists across steps, a larger surface of side effects to assert against, and a trajectory measured in hours rather than seconds.

### **Dimensions of evaluation**

These dimensions are orthogonal, not alternatives to choose between. Most mature systems use several at once.

**Pointwise and pairwise.** Pairwise evals, also called side-by-side or SxS, show a rater two outputs for the same input and ask which one is better. LMArena is the well-known public example. The signal is high quality, but it needs human raters, so running it on every change is impractical. Pointwise evals score a single output against defined criteria and are what you want in the development loop.

**Response and trajectory.** Response evals grade the final output. Trajectory evals grade the path taken. If the agent’s job is to answer a question, the path may not matter. If its job is to issue a refund, the path is key and must be evaluated.

**Offline and online.** Offline evals run against a curated dataset in a controlled environment and catch defects before release. Online evals score a sample of production traffic, on the assumption that no offline dataset can anticipate what real users will do. You need both, and online evals are how the offline dataset stays representative (more on that later). Privacy and data residency rules in your operating region can restrict online evaluation, so confirm those before you start collecting traces.

**Ad hoc and scheduled.** Ad hoc runs work like unit tests during development: change a prompt, run a subset of examples, look at the result. Google ADK Web provides a way to run evals on your agents while in development. Scheduled runs, executed nightly and/or before each release, work like integration tests over the full dataset.

**Component and end-to-end.** In a multi-agent system you can evaluate individual subagents in isolation, such as a retriever, a router or a single tool wrapper. Component runs are much cheaper, have less variance, and localise failures. End-to-end runs take time, so reserve them for critical user journeys and cover the underlying subsystems with component evals.

### **Architecture of an eval harness**

The pipeline: eval set in, verdicts out, failures returned to the eval set.

The pipeline: eval set in, verdicts out, failures returned to the eval set.

The pipeline: eval set in, verdicts out, failures returned to the eval set.

A harness has six parts:

- The eval set, which holds inputs and optionally the expected outputs;
- The environment, which is some mix of mocked tools, a sandbox or a dedicated test account;
- Agent under test, pinned to a specific version of every dependency;
- Traces produced by running it;
- Autoraters that judge those traces and return verdicts;
- Aggregation layer that produces metric scores and reporting.

### **The eval set**

An eval set is a collection of examples that start an agent run and carry the context needed to execute it.

Examples come from four places, roughly in order of value.

**Production traces and bug reports are worth more than the rest combined.** A failure you have already observed is a valuable test case for future runs, especially when it happens in production.

**Synthetic generation** has an LLM produce examples from critical user journeys, known defects and existing traces. It is fast and covers the long tail, but the examples are less realistic. It is most useful when you need to measure the quality of a new agent before it reaches production.

**Open-source benchmarks** such as SWE-bench for coding are well established for common problem domains. They calibrate your agent against the wider field, and they are rarely sufficient for a specific product.

**Vendor-supplied datasets** come from specialist agencies that build eval sets and test environments to your specification. This is the most expensive option, and often the fastest route to coverage when you have budget and no data.

A golden dataset is an eval set that also carries expected outputs: the trajectory, the tool calls, the final response, or the resulting state. Autoraters compare observed behaviour against those expectations.

Coverage is the hardest part of building an eval set. A few practices cut the cost of getting there.

- **Version the eval set as code.** A score only means something against the same version of the dataset, and changing the dataset and the agent in the same commit gives you a number you cannot interpret.
- **Stratify examples and report per slice.** A single aggregate figure can hide a bad regression in a small but commercially important segment of traffic. Generally, slicing does not work on small datasets of less than a 100 examples, where confidence intervals are very wide to be useful.
- **Maintain a hold-out set.** Tuning agent prompts against every example you have is overfitting to the eval set. A hold-out slice helps you independently validate your fixes.
- **Refresh the dataset periodically.** Products change, models get swapped out, and user behaviour drifts. A stale eval set keeps reporting success while production quality drops.

### **The environment**

Agents need something to call. The options form a fidelity spectrum, and the choices come with trade-offs.

**Static mocks** are fake tool responses matched on arguments and written for each test. They are cheap, deterministic and isolated from downstream instability. But they will not catch schema drift in the underlying API.

**Recorded and replayed responses** are captured from live systems and replayed inside the harness. More realistic, but they go stale and need periodic replays. This practice is prevalent in Integration tests.

**A sandboxed environment** is a working stack with seeded data. Good fidelity at a meaningful maintenance cost.

**A dedicated test account against live systems** gives the highest fidelity and costs the most to set up. Every eval example needs its preconditions to exist in the account, which in practice makes this slow and expensive. This is also the option with highest flakiness as system reliability also comes into picture.

For most teams the right combination is mocks or recorded responses inside the eval loop plus one of the other 2 options outside it. Mocks keep evals fast and repeatable, and the higher-fidelity option catches integration problems in a prod-like environment.

### **Execution and traces**

Running the agent over every example produces the artifacts you grade: responses, intermediate reasoning, tool calls and their results, and state changes. Two requirements apply here.

**Pin every variable that is not under test.** Model version, system prompt version, tool schemas, eval set version, judge configuration. If two of them change at once, you cannot attribute the delta in eval metrics to either. Culprits are intuitive in single and smaller agents but versioning becomes non-negotiable in complex and multi-agent systems.

**Run each example several times.** Agents are stochastic, so a single run is a sample rather than a measurement (more on this below).

Instrument the agent with a standard tracing format like OpenTelemetry GenAI semantic conventions. The same trace schema then serves both the offline harness and production observability, which is the cheapest route to online evaluation available as both systems reuse the eval harness.

### **Autoraters**

Three kinds of autorater exist, and a complete system uses all three.

**Rule-based autoraters** are ordinary code. Use them wherever the correct outcome is deterministic: a tool was called with specific arguments, a database record holds a particular value, the response contains one of a set of approved disclaimers, or no issue_refund call happened before inspection. They are cheap, fast, and they do not drift.

Established text metrics such as BLEU and ROUGE belong here too, but they give weak signal for agents and should be kept to narrow, string-comparable outputs like translation and summarisation.

**LLM judges** cover the criteria that need judgment: groundedness, trajectory coherence, tone, and policy adherence that cannot be reduced to a string match. Custom judges are rarely necessary at the start. The judges published by the SDKs and by the research community are a reasonable starting point and can be tuned later.

One of the more reliable judge designs uses binary rubrics, breaking each criterion into atomic yes/no assertions instead of asking for an ordinal score. “Did the agent call get_order_details before generate_return_label?” is a well-defined question. “Rate trajectory quality from 1 to 5” is not, and it could produce an unstable number that changes every run. Rubrics can be static, defined in the eval example or the judge prompt, or derived from the trajectory at runtime.

Judges are usually also asked for a justification and for the failing step in the trajectory. Making the judge cite the point in the trajectory that decided its verdict cuts down unsupported conclusions, and those citations are useful when you go looking for things to fix in the agent.

**Human review** is the smallest component by volume and the most important, because it is what validates the other two. Validate the judge against human review before you rely on its output. Evals with a misaligned autorater provide misleading signals that don’t add up with real user experience.

### **Single-turn and multi-turn evals**

Single-turn evals grade one exchange, either a final response or one step of a trajectory. They are easier to build and score. Most agent evals start with single-turn evals.

Multi-turn evals are needed for conversational agents, and they need a simulated user: an LLM given a persona, an objective, and a set of facts it will only reveal when asked. That is what makes it possible to test whether the agent asks sensible clarifying questions, because there is now a counterparty to answer them.

Scoring moves up to the conversation level: whether the user’s objective was met, how many turns it took, whether the agent recovered from ambiguity or contradiction, and whether policy held across the whole conversation rather than just the opening message.

When you use simulated users, cap turn counts, define stopping conditions explicitly, and read a sample of transcripts by hand to check if the simulated users still resemble real ones.

### **Evaluating agents reliably**

Run the same eval example twice and you can get two different trajectories.

Report intervals rather than point estimates. Pass rates are normally reported as a mean over several runs with a confidence interval and standard deviation to get a holistic picture of the agent’s performance and quality.

Run each example k times and report the pass rate across runs. For journeys where reliability is itself the requirement, particularly anything financial or irreversible, also report pass^k, the proportion of examples that pass on every attempt. That figure tracks what users actually experience far more closely than a mean does. Its more lenient counterpart is pass@k, the proportion of examples that passed at least once in k attempts.

Separate infrastructure failures from agent failures. Timeouts, quota exhaustion and errors returned by a mock server are not quality signals, and they should be tracked on their own. Otherwise part of what your metric measures is the stability of your test infrastructure.

### **Integrating evals into the release process**

Evals belong in CI in tiers.

- **Smoke suite.** Runs on every change, finishes in minutes, and covers a small stratified sample plus all the rule-based checks. It catches obvious breakage during development.
- **Nightly full suite.** Runs the whole eval set with k runs per example.
- **Release gate.** Runs the full set and measures cost and latency. It needs to meet defined thresholds against the current production baseline.

### **Cost management**

Evals are not cheap. Generating trajectories and LLM judges both burn tokens. Parallelised, a full eval run can exhaust your model quota and put load on the backend.

Two things have helped me in the past:

- Move as many checks as you can into rule-based autoraters, which are effectively free.
- Deduplicate the eval set, because near-identical examples cost as much as distinct ones and tell you less.

### **Conclusion**

Agents fail differently from models, so they need different measurements.

1. **The unit of evaluation is the trajectory, not the response.** Grading the final response alone is not enough signal, which is why agentic evals add trajectory, tool use, side effects and efficiency on top of the criteria inherited from LLM evaluation.
2. **The eval set determines the value of everything downstream.** It is okay to start with manual debugging traces and synthetic sources. But production traces and bug reports should be integrated later to keep them realistic. Open-source benchmarks and vendor-supplied datasets are also an option. Version the dataset as code and keep a hold-out slice so you can measure improvement without overfitting.
3. **Environment fidelity is expensive, so spend it carefully.** Mocks, replayed responses, sandboxes and dedicated test accounts trade cost against realism, and most teams want mocks inside the eval loop with a higher-fidelity option outside it.
4. **A complete autorater stack needs rule-based checks, LLM judges and human review.** Deterministic outcomes belong in ordinary code, which is fast, cheap and does not drift. Judgment calls belong to LLM judges, which work best as binary rubrics that have to cite the trajectory step behind the verdict. Human review is the smallest piece by volume and the one that validates the other two before you rely on either.
5. **Agents are stochastic, so a single score is a sample rather than a measurement.** Run each example k times and report pass rates as means with confidence intervals, and track infrastructure failures separately from agent failures.