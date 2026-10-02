# Third-Party Notices

ReactiveGraph is distributed under the MIT License (see [`LICENSE`](LICENSE)).

Parts of this repository are derived from third-party open-source projects.
Their licenses and copyright notices are reproduced below, as required by the
MIT License. Files that contain derived code carry a short header pointing
back to this document.

---

## LangGraph / LangChain

`python/reactivegraph/reactivegraph/tool_node.py`,
`python/reactivegraph/reactivegraph/tools.py`,
`python/reactivegraph/reactivegraph/messages.py`,
`python/reactivegraph/reactivegraph/message_utils.py`,
`python/reactivegraph/reactivegraph/middleware.py`,
`python/reactivegraph/reactivegraph/todo.py`,
`python/reactivegraph/reactivegraph/summarization.py`,
`python/reactivegraph/reactivegraph/channels/__init__.py`,
`python/reactivegraph/reactivegraph/checkpoint.py`,
`python/reactivegraph/reactivegraph/runtime.py`,
`python/reactivegraph/reactivegraph/message_state.py`,
`python/reactivegraph/reactivegraph/types.py`,
`python/reactivegraph/reactivegraph/store.py` and
`python/reactivegraph/reactivegraph/warnings.py` adapt functions, class
structures and behavioural contracts from:

- [`langchain-ai/langgraph`](https://github.com/langchain-ai/langgraph)
  (`langgraph`, `langgraph-checkpoint`, `langgraph-prebuilt`)
- [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain)
  (`langchain-core`, `langchain`)

`todo.py` ports the `TodoListMiddleware` prompt text and behaviour, and
`summarization.py` ports `SummarizationMiddleware`, both from the `langchain`
distribution (`langchain/agents/middleware/`).

The adaptation reimplements the observable surface (signatures, semantics,
error behaviour) so that hosts written against LangGraph/LangChain keep
working on the ReactiveGraph engine. Reference versions:

| Upstream package | Version used as reference |
| --- | --- |
| `langgraph` | 1.2.11 |
| `langgraph-checkpoint` | 4.2.0 |
| `langgraph-prebuilt` | 1.1.0 |
| `langchain-core` | 1.6.4 |
| `langchain` | 1.3.14 |

### MIT License

```
MIT License

Copyright (c) 2024 LangChain, Inc.
Copyright (c) LangChain, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## uuid6-python

`python/reactivegraph/reactivegraph/checkpoint.py` implements UUID version 6
generation adapted from
[`oittaa/uuid6-python`](https://github.com/oittaa/uuid6-python), the same
upstream that LangGraph bundles as `langgraph/checkpoint/base/id.py`.

### MIT License

```
MIT License

Copyright (c) 2021 oittaa

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Scope note

The TypeScript packages under `packages/` are an independent implementation of
the RGP/1 protocol and the reactive engine; they do not embed LangChain or
LangGraph source code. `benchmarks/differential/` compares against upstream
packages by importing them, and does not vendor them.
