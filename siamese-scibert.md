```mermaid
graph TD
    subgraph Input Layer
        A[Abstract A]
        B[Abstract B]
    end

    subgraph Encoder Layer (Shared Weights)
        SciBERTA(SciBERT Model)
        SciBERTB(SciBERT Model)
    end

    subgraph Pooling Layer
        MeanPoolA(Mean Pooling)
        MeanPoolB(Mean Pooling)
    end

    subgraph Feature Engineering Layer
        VecXA(Vector X)
        VecYB(Vector Y)
        AbsDiff("|X - Y|")
        ElementProd("X * Y")
        Concat[Concatenation]
    end

    subgraph Classification Head
        Linear(Linear Layer)
        Output[Output Logits (0 or 1)]
    end

    A --> SciBERTA
    B --> SciBERTB

    SciBERTA --> MeanPoolA
    SciBERTB --> MeanPoolB

    MeanPoolA --> VecXA
    MeanPoolB --> VecYB

    VecXA --> AbsDiff
    VecYB --> AbsDiff

    VecXA --> ElementProd
    VecYB --> ElementProd

    VecXA --> Concat
    VecYB --> Concat
    AbsDiff --> Concat
    ElementProd --> Concat

    Concat --> Linear
    Linear --> Output

    style SciBERTB fill:#fff,stroke:#333,stroke-width:2px,stroke-dasharray: 5 5
    linkStyle 4 stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    linkStyle 5 stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    linkStyle 6 stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
    linkStyle 7 stroke:#333,stroke-width:2px,stroke-dasharray: 5 5;
```