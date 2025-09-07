# Blog Content Management via GitHub

This portfolio website now supports easy blog content management through GitHub using Markdown format. You can add, edit, and update blog posts without dealing with HTML code.

## How to Edit Blogs

### 1. Locate the Blog Data File
The blog content is stored in: `src/data/markdownBlogs.ts`

### 2. Edit Existing Blogs
To update existing blog content:
1. Navigate to `src/data/markdownBlogs.ts` in your GitHub repository
2. Click the "Edit" button (pencil icon)
3. Find the blog post you want to edit by its `id`
4. Modify the `markdownContent` field using standard Markdown syntax
5. Commit your changes

### 3. Add New Blogs
To add a new blog post:
1. Open `src/data/markdownBlogs.ts`
2. Add a new object to the `markdownBlogs` array:

```typescript
{
  id: "unique-blog-id",
  title: "Your Blog Title",
  description: "Short description for the blog card",
  date: "2024-03-20",
  tags: ["Tag1", "Tag2", "Tag3"],
  readTime: "15 min read",
  category: "Data Analysis", // or other categories
  featured: false, // set to true for featured posts
  markdownContent: `
# Your Blog Title

Your blog content goes here using Markdown syntax...

## Section Heading

- Bullet point 1
- Bullet point 2

### Code Examples

\`\`\`python
# Python code example
import pandas as pd
df = pd.read_csv('data.csv')
\`\`\`

**Bold text** and *italic text* are supported.
  `
}
```

## Supported Markdown Features

The blog system supports all standard Markdown features:

### Text Formatting
- **Bold text**: `**bold**` or `__bold__`
- *Italic text*: `*italic*` or `_italic_`
- `Inline code`: \`code\`

### Headings
```markdown
# H1 Heading
## H2 Heading
### H3 Heading
#### H4 Heading
```

### Lists
```markdown
- Bullet point 1
- Bullet point 2

1. Numbered list item 1
2. Numbered list item 2
```

### Code Blocks with Syntax Highlighting
```markdown
\`\`\`python
# Python example
def hello_world():
    print("Hello, World!")
\`\`\`

\`\`\`sql
-- SQL example
SELECT * FROM users WHERE active = true;
\`\`\`

\`\`\`excel
// Excel formula example
=VLOOKUP(A2, Table1, 2, FALSE)
\`\`\`
```

### Links and Images
```markdown
[Link text](https://example.com)
![Image alt text](image-url.jpg)
```

### Tables
```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
```

### Blockquotes
```markdown
> This is a blockquote
> It can span multiple lines
```

## Available Categories

Current blog categories include:
- Data Analysis
- Machine Learning
- Deep Learning
- Generative AI
- AI Agents
- Agentic AI
- MCP

## Best Practices

1. **Use descriptive IDs**: Make blog IDs lowercase with hyphens (e.g., `"advanced-sql-techniques"`)
2. **Keep descriptions concise**: Aim for 1-2 sentences in the description field
3. **Use relevant tags**: Add 3-6 relevant tags to help with categorization
4. **Set appropriate read times**: Estimate based on content length
5. **Use proper Markdown formatting**: This ensures consistent styling across the site
6. **Test locally**: If you have the development environment, test your changes locally before pushing

## Automatic Features

The blog system automatically handles:
- ✅ Syntax highlighting for code blocks
- ✅ Responsive design for mobile and desktop
- ✅ SEO optimization
- ✅ Dark/light theme support
- ✅ Social sharing functionality
- ✅ Consistent typography and spacing

## Quick Editing Workflow

1. **GitHub Web Interface**: Edit directly in your browser
2. **Git Clone**: Clone the repository and edit locally
3. **GitHub Desktop**: Use GitHub's desktop app for easier file management

After pushing changes to GitHub, your blog updates will be automatically deployed to your live site!