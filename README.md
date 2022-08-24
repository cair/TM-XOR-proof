# Convergence-of-Tsetlin-Machine-for-Fundamental-Digital-Operations-the-XOR-Case

The Tsetlin Machine (TM) is a novel machine learning algorithm with several distinct properties, including transparent inference/learning and hardware-near building blocks. Although numerous papers explore the TM empirically, many of its properties have not yet been analyzed mathematically. In this article, we analyze the convergence of the TM when input is non-linearly related to output by the XOR-operator. Our analysis reveals that the TM, with just two conjunctive clauses, can converge surely to reproducing XOR, learning from training data over an infinite time horizon. Furthermore, the analysis shows how the hyper-parameter T guides clause construction so that the clauses capture the distinct sub-patterns in the data. Our analysis of convergence for XOR thus lays the foundation for analyzing other more complex logical expressions. These analyses altogether, from a mathematical perspective, provide new insights on why TMs have obtained state-of-the-art performance on several pattern recognition problems.

```ruby
require 'redcarpet'
markdown = Redcarpet.new("Hello World!")
puts markdown.to_html
```
