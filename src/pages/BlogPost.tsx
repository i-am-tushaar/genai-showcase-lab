import { useParams, Link } from "react-router-dom";
import { ArrowLeft, Calendar, Clock, Tag, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { blogPosts } from "@/data/blogData";
import { markdownBlogs } from "@/data/markdownBlogs";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";

const BlogPost = () => {
  const { id } = useParams<{ id: string }>();
  
  // First check markdown blogs, then fallback to regular blogs
  const markdownBlog = markdownBlogs.find(post => post.id === id);
  const blogPost = blogPosts.find(post => post.id === id);
  const post = markdownBlog || blogPost;

  if (!post) {
    return (
      <div className="min-h-screen pt-20 flex items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold mb-4">Blog Post Not Found</h1>
          <p className="text-muted-foreground mb-8">The blog post you're looking for doesn't exist.</p>
          <Button asChild>
            <Link to="/blogs">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Blogs
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: post.title,
        text: post.description,
        url: window.location.href,
      });
    } else {
      // Fallback to copying URL
      navigator.clipboard.writeText(window.location.href);
      // You could add a toast notification here
    }
  };

  return (
    <div className="min-h-screen pt-20">
      {/* Back Button */}
      <div className="container mx-auto px-4 py-8">
        <Button variant="ghost" asChild className="mb-8">
          <Link to="/blogs">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Blogs
          </Link>
        </Button>
      </div>

      {/* Article Header */}
      <article className="container mx-auto px-4 max-w-4xl">
        <header className="mb-12">
          <div className="flex items-center gap-2 text-sm text-muted-foreground mb-6">
            <Calendar className="h-4 w-4" />
            <span>{new Date(post.date).toLocaleDateString('en-US', { 
              year: 'numeric', 
              month: 'long', 
              day: 'numeric' 
            })}</span>
            <Clock className="h-4 w-4 ml-4" />
            <span>{post.readTime}</span>
          </div>
          
          <h1 className="text-4xl md:text-5xl font-bold mb-6 leading-tight">
            {post.title}
          </h1>
          
          <p className="text-xl text-muted-foreground leading-relaxed mb-8">
            {post.description}
          </p>
          
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-2 flex-wrap">
              <Tag className="h-4 w-4 text-muted-foreground" />
              {post.tags.map((tag, index) => (
                <Badge key={index} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>
            
            <Button variant="outline" size="sm" onClick={handleShare}>
              <Share2 className="h-4 w-4 mr-2" />
              Share
            </Button>
          </div>
        </header>

        {/* Article Content */}
        <div className="prose prose-lg max-w-none dark:prose-invert">
          {markdownBlog ? (
            <MarkdownRenderer 
              content={markdownBlog.markdownContent}
              className="mt-8"
            />
          ) : (
            <div 
              className="markdown-content"
              dangerouslySetInnerHTML={{ 
                __html: blogPost?.content || ''
              }}
            />
          )}
        </div>

        {/* Related Posts or CTA */}
        <footer className="mt-16 pt-8 border-t border-border">
          <Card className="p-6 gradient-bg">
            <div className="text-center">
              <h3 className="text-2xl font-bold mb-4">Enjoyed this article?</h3>
              <p className="text-muted-foreground mb-6">
                Get more insights on AI engineering, MLOps, and cutting-edge AI technologies
              </p>
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <Button asChild>
                  <Link to="/blogs">Read More Articles</Link>
                </Button>
                <Button variant="outline" asChild>
                  <Link to="/contact">Get In Touch</Link>
                </Button>
              </div>
            </div>
          </Card>
        </footer>
      </article>
    </div>
  );
};

export default BlogPost;