import { Link } from "react-router-dom";
import { Calendar, Clock, Tag, Star } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { blogPosts, categories } from "@/data/blogData";

const Blogs = () => {
  const featuredPosts = blogPosts.filter(post => post.featured);
  const categorizedPosts = categories.reduce((acc, category) => {
    acc[category] = blogPosts.filter(post => post.category === category && !post.featured);
    return acc;
  }, {} as Record<string, typeof blogPosts>);

  return (
    <div className="min-h-screen pt-20">
      {/* Header Section */}
      <section className="py-20 px-4 gradient-bg">
        <div className="container mx-auto text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-6">
            <span className="text-gradient">AI & Data Science Blog</span>
          </h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Comprehensive guides and insights across Data Analysis, Machine Learning, Deep Learning, Generative AI, and Agentic AI systems
          </p>
        </div>
      </section>

      {/* Featured Posts */}
      {featuredPosts.length > 0 && (
        <section className="py-16 px-4 bg-muted/30">
          <div className="container mx-auto">
            <div className="flex items-center gap-2 mb-8">
              <Star className="h-6 w-6 text-primary" />
              <h2 className="text-2xl md:text-3xl font-bold">Featured Articles</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              {featuredPosts.map((post) => (
                <Card 
                  key={post.id} 
                  className="group hover:scale-105 transition-all duration-300 bg-card/50 backdrop-blur-sm border-border/50 cursor-pointer h-full flex flex-col relative overflow-hidden"
                >
                  <div className="absolute top-4 right-4 z-10">
                    <Badge variant="secondary" className="bg-primary text-primary-foreground">
                      Featured
                    </Badge>
                  </div>
                  <Link to={`/blogs/${post.id}`} className="flex flex-col h-full">
                    <CardHeader className="flex-grow">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                        <Calendar className="h-4 w-4" />
                        <span>{new Date(post.date).toLocaleDateString('en-US', { 
                          year: 'numeric', 
                          month: 'long', 
                          day: 'numeric' 
                        })}</span>
                        <Clock className="h-4 w-4 ml-2" />
                        <span>{post.readTime}</span>
                      </div>
                      
                      <CardTitle className="text-xl group-hover:text-primary transition-colors mb-3 line-clamp-2">
                        {post.title}
                      </CardTitle>
                      
                      <CardDescription className="text-base leading-relaxed line-clamp-3">
                        {post.description}
                      </CardDescription>
                    </CardHeader>
                    
                    <CardContent className="pt-0 mt-auto">
                      <div className="flex items-center gap-2 flex-wrap">
                        <Tag className="h-4 w-4 text-muted-foreground" />
                        {post.tags.slice(0, 3).map((tag, index) => (
                          <Badge 
                            key={index} 
                            variant="secondary" 
                            className="text-xs"
                          >
                            {tag}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Link>
                </Card>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Categorized Blog Posts */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          {categories.map((category) => {
            const categoryPosts = categorizedPosts[category];
            if (categoryPosts.length === 0) return null;
            
            return (
              <div key={category} className="mb-16">
                <h2 className="text-2xl md:text-3xl font-bold mb-8 text-gradient">
                  {category}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {categoryPosts.map((post) => (
                    <Card 
                      key={post.id} 
                      className="group hover:scale-105 transition-all duration-300 bg-card/50 backdrop-blur-sm border-border/50 cursor-pointer h-full flex flex-col"
                    >
                      <Link to={`/blogs/${post.id}`} className="flex flex-col h-full">
                        <CardHeader className="flex-grow">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                            <Calendar className="h-4 w-4" />
                            <span>{new Date(post.date).toLocaleDateString('en-US', { 
                              year: 'numeric', 
                              month: 'long', 
                              day: 'numeric' 
                            })}</span>
                            <Clock className="h-4 w-4 ml-2" />
                            <span>{post.readTime}</span>
                          </div>
                          
                          <CardTitle className="text-xl group-hover:text-primary transition-colors mb-3 line-clamp-2">
                            {post.title}
                          </CardTitle>
                          
                          <CardDescription className="text-base leading-relaxed line-clamp-3">
                            {post.description}
                          </CardDescription>
                        </CardHeader>
                        
                        <CardContent className="pt-0 mt-auto">
                          <div className="flex items-center gap-2 flex-wrap">
                            <Tag className="h-4 w-4 text-muted-foreground" />
                            {post.tags.slice(0, 3).map((tag, index) => (
                              <Badge 
                                key={index} 
                                variant="secondary" 
                                className="text-xs"
                              >
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        </CardContent>
                      </Link>
                    </Card>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Newsletter CTA Section */}
      <section className="py-20 px-4 gradient-bg">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Stay Updated with Latest AI Insights
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Get notified when I publish new articles about AI engineering, MLOps, and cutting-edge AI technologies
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center max-w-md mx-auto">
            <input
              type="email"
              placeholder="Enter your email"
              className="flex-1 px-4 py-3 rounded-md border border-input bg-background text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
            />
            <button className="px-6 py-3 gradient-primary text-primary-foreground rounded-md font-medium hover:opacity-90 transition-opacity">
              Subscribe
            </button>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Blogs;