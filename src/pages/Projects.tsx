import { ExternalLink, Github, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import project1 from "@/assets/project-1.jpg";
import project2 from "@/assets/project-2.jpg";
import project3 from "@/assets/project-3.jpg";

const Projects = () => {
  const projects = [
    {
      id: 1,
      title: "End-to-End LLMOps Platform",
      description:
        "Advanced computer vision system using transformer architectures for real-time object detection and analysis. Achieved 98.5% accuracy on custom datasets.",
      image: project1,
      tags: ["PyTorch", "Computer Vision", "Transformers", "Docker"],
      featured: true,
      github: "#",
      demo: "#",
      metrics: ["98.5% Accuracy", "30ms Inference", "50K+ Images Processed"],
    },
    {
      id: 2,
      title: "MLOps Pipeline Platform",
      description:
        "End-to-end MLOps platform enabling automated model training, validation, deployment, and monitoring with A/B testing capabilities.",
      image: project2,
      tags: ["Kubernetes", "MLflow", "Airflow", "AWS"],
      featured: true,
      github: "#",
      demo: "#",
      metrics: ["15+ Models Deployed", "99.9% Uptime", "40% Cost Reduction"],
    },
    {
      id: 3,
      title: "Intelligent Agent Framework",
      description:
        "Multi-agent system for autonomous task execution with natural language understanding and dynamic workflow orchestration.",
      image: project3,
      tags: ["LangChain", "Vector DB", "FastAPI", "Redis"],
      featured: false,
      github: "#",
      demo: "#",
      metrics: ["85% Task Success", "5sec Response", "Multi-Modal Input"],
    },
    {
      id: 4,
      title: "LLM Fine-tuning Pipeline",
      description:
        "Scalable pipeline for fine-tuning large language models on domain-specific data with parameter-efficient techniques.",
      image: project1,
      tags: ["Transformers", "LoRA", "PEFT", "Distributed Training"],
      featured: false,
      github: "#",
      demo: "#",
      metrics: ["70% Performance Gain", "90% Memory Efficient", "Multi-GPU Training"],
    },
    {
      id: 5,
      title: "Real-time Recommendation Engine",
      description:
        "Hybrid recommendation system combining collaborative filtering and deep learning for personalized content delivery.",
      image: project2,
      tags: ["TensorFlow", "Redis", "Apache Kafka", "PostgreSQL"],
      featured: false,
      github: "#",
      demo: "#",
      metrics: ["35% CTR Improvement", "<100ms Response", "1M+ Users Served"],
    },
    {
      id: 6,
      title: "Automated Data Labeling Tool",
      description:
        "AI-powered data labeling platform using active learning and weak supervision to reduce manual annotation effort.",
      image: project3,
      tags: ["Active Learning", "Weak Supervision", "React", "FastAPI"],
      featured: false,
      github: "#",
      demo: "#",
      metrics: ["80% Label Reduction", "95% Quality Score", "10x Faster Labeling"],
    },
  ];

  const featuredProjects = projects.filter((p) => p.featured);
  const otherProjects = projects.filter((p) => !p.featured);

  return (
    <div className="min-h-screen pt-20 px-4">
      <div className="container mx-auto">
        {/* Header */}
        <section className="py-20 text-center">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">My Projects</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            A showcase of AI and ML solutions I've built, from research prototypes to production systems
          </p>
        </section>

        {/* Featured Projects */}
        <section className="py-16">
          <h2 className="text-3xl font-bold mb-12 flex items-center gap-2">
            <Star className="h-8 w-8 text-primary" />
            Featured Projects
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {featuredProjects.map((project) => (
              <Card
                key={project.id}
                className="overflow-hidden group hover:scale-[1.02] transition-all duration-300 bg-card/50 backdrop-blur-sm"
              >
                <div className="aspect-video overflow-hidden">
                  <img
                    src={project.image}
                    alt={project.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <div className="p-6">
                  <h3 className="text-2xl font-bold mb-2">{project.title}</h3>
                  <p className="text-muted-foreground mb-4">{project.description}</p>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {project.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>

                  <div className="grid grid-cols-3 gap-2 mb-4">
                    {project.metrics.map((metric, index) => (
                      <div key={index} className="text-center">
                        <p className="text-sm font-semibold text-primary">{metric}</p>
                      </div>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <a href={project.github} target="_blank" rel="noopener noreferrer">
                        <Github className="h-4 w-4 mr-2" />
                        Code
                      </a>
                    </Button>
                    <Button size="sm" asChild>
                      <a href={project.demo} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="h-4 w-4 mr-2" />
                        Demo
                      </a>
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Other Projects */}
        <section className="py-16">
          <h2 className="text-3xl font-bold mb-12">More Projects</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
            {otherProjects.map((project) => (
              <Card
                key={project.id}
                className="overflow-hidden group hover:scale-105 transition-all duration-300 bg-card/50 backdrop-blur-sm"
              >
                <div className="aspect-video overflow-hidden">
                  <img
                    src={project.image}
                    alt={project.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                  />
                </div>
                <div className="p-4">
                  <h3 className="text-lg font-bold mb-2">{project.title}</h3>
                  <p className="text-muted-foreground text-sm mb-3 line-clamp-2">
                    {project.description}
                  </p>

                  <div className="flex flex-wrap gap-1 mb-3">
                    {project.tags.slice(0, 3).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>

                  <div className="flex gap-2">
                    <Button variant="outline" size="sm" asChild>
                      <a href={project.github} target="_blank" rel="noopener noreferrer">
                        <Github className="h-4 w-4" />
                      </a>
                    </Button>
                    <Button size="sm" asChild>
                      <a href={project.demo} target="_blank" rel="noopener noreferrer">
                        <ExternalLink className="h-4 w-4" />
                      </a>
                    </Button>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default Projects;
