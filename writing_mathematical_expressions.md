Writing mathematical expressions
Use Markdown to display mathematical expressions on GitHub.

Who can use this feature?
Markdown can be used in the GitHub web interface.

In this article
About writing mathematical expressions
To enable clear communication of mathematical expressions, GitHub supports LaTeX formatted math within Markdown. For more information, see LaTeX/Mathematics in Wikibooks.

GitHub's math rendering capability uses MathJax; an open source, JavaScript-based display engine. MathJax supports a wide range of LaTeX macros, and several useful accessibility extensions. For more information, see the MathJax documentation and the MathJax Accessibility Extensions Documentation.

Mathematical expressions rendering is available in GitHub Issues, GitHub Discussions, pull requests, wikis, and Markdown files.

Writing inline expressions
There are two options for delimiting a math expression inline with your text. You can either surround the expression with dollar symbols ($), or start the expression with $` and end it with `$. The latter syntax is useful when the expression you are writing contains characters that overlap with markdown syntax. For more information, see Basic writing and formatting syntax.

This sentence uses `$` delimiters to show math inline: $\sqrt{3x-1}+(1+x)^2$
Screenshot of rendered Markdown showing an inline mathematical expression: the square root of 3x minus 1 plus (1 plus x) squared.

This sentence uses $\` and \`$ delimiters to show math inline: $`\sqrt{3x-1}+(1+x)^2`$
Screenshot of rendered Markdown showing an inline mathematical expression with backtick syntax: the square root of 3x minus 1 plus (1 plus x) squared.

Writing expressions as blocks
To add a math expression as a block, start a new line and delimit the expression with two dollar symbols $$.

Tip

If you're writing in an .md file, you will need to use specific formatting to create a line break, such as ending the line with a backslash as shown in the example below. For more information on line breaks in Markdown, see Basic writing and formatting syntax.

**The Cauchy-Schwarz Inequality**\
$$\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)$$
Screenshot of rendered Markdown showing a complex equation. Bold text reads "The Cauchy-Schwarz Inequality" above the formula for the inequality.

Alternatively, you can use the ```math code block syntax to display a math expression as a block. With this syntax, you don't need to use $$ delimiters. The following will render the same as above:

**The Cauchy-Schwarz Inequality**

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```
Writing dollar signs in line with and within mathematical expressions
To display a dollar sign as a character in the same line as a mathematical expression, you need to escape the non-delimiter $ to ensure the line renders correctly.

Within a math expression, add a \ symbol before the explicit $.

This expression uses `\$` to display a dollar sign: $`\sqrt{\$4}`$
Screenshot of rendered Markdown showing how a backslash before a dollar sign displays the sign as part of a mathematical expression.

Outside a math expression, but on the same line, use span tags around the explicit $.