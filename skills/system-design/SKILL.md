---
name: system-design
version: 0.1.0
description: Draw system-design diagrams in ASCII with proper notation.
triggers: [diagram, system-design, architecture, flowchart, design, url-shortener, components]
tools: []
category: software-development
---


# System Design Skill

## 1. Understand the Problem
- Identify users, core workflows, scope, constraints, and assumptions.
- Ask clarifying questions only when missing information materially affects the design.
- Never silently invent requirements.

## 2. Requirements
List before designing:
- Functional requirements and priorities.
- Non-functional requirements: latency, throughput, availability, consistency, scalability, security, and durability.
- Constraints, assumptions, and capacity estimates when relevant.

## 3. Component Inventory
**High-level:** List major services, external systems, and data stores with their responsibilities.

**Low-level:** Identify relevant internal modules, APIs, data models, queues, workers, caches, and processing stages.

Explain each component's purpose. Avoid unnecessary decomposition.

## 4. Architecture Decisions
Explain important choices, alternatives, and trade-offs. Consider communication protocols, databases, caching, consistency, partitioning, and deployment when relevant.

Prefer the simplest architecture that satisfies the requirements. Never introduce technologies without justification.

## 5. Design Depth
Adapt detail to the request:
- Quick: Requirements, components, diagram, execution example.
- Interview: Capacity estimates, architecture, trade-offs, bottlenecks.
- Production: APIs, data models, failure handling, security, observability, deployment, and recovery.

## 6. ASCII Diagram Rules
- Use ASCII only: +---+ boxes, --> arrows, [db: name] for storage, (queue: name) for queues, and o for users.
- Use one primary component per box.
- Label important connections with protocols, requests, events, or data.
- Keep diagrams terminal-width friendly; prefer top-down layouts.
- Split complex designs into multiple meaningful views.
- Use consistent component names throughout.
- Never use Mermaid, drawing code, or Markdown fences.

## 7. Architecture Views
Include relevant views:
- High-level architecture.
- Low-level component design.
- Request or event flow.
- Data flow and storage architecture.

Avoid redundant diagrams.

## 8. Execution Walkthrough
After the diagram, trace one concrete operation end to end. Explain the request path, processing, storage interactions, response, and relevant error handling. Follow the diagram exactly.

## 9. Production Readiness
Assess relevant failure modes, timeouts, retries, idempotency, consistency, bottlenecks, scaling, security, logging, metrics, and tracing. Separate essential requirements from future improvements.

## 10. Validation
Before responding, verify:
- Requirements map to components.
- Components have clear responsibilities.
- Arrows and data flows are complete.
- Storage and external dependencies are explicit.
- No unexplained components or dangling connections exist.
- The execution example matches the diagram.
- Complexity is justified.

## 11. Default Output Order
1. Problem and assumptions
2. Requirements and constraints
3. High-level components
4. Low-level components
5. Architecture decisions and trade-offs
6. ASCII diagrams
7. End-to-end example
8. Production concerns and limitations

Omit irrelevant sections for simple requests. Use plain English, distinguish assumptions from facts, and prioritize correctness over complexity.

## Design Mode Selection
Before designing, determine the required depth. If the user has not specified it, ask them to choose:

1. **Conceptual:** Basic components, simplified ASCII diagram, and a brief execution flow.
2. **System Design / Interview:** Requirements, capacity estimates, high-level and low-level components, architecture trade-offs, scaling, and bottlenecks.
3. **Production Grade:** Detailed architecture, APIs, data models, infrastructure, failure handling, security, observability, deployment, and recovery.

Do not ask again if the user's request clearly establishes the desired depth. Adapt the output to the selected mode without including unnecessary sections.