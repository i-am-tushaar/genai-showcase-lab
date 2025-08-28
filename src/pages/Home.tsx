import { ArrowRight, Brain, Code, Zap, Cloud, Database, Layers } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import heroImage from "@/assets/hero-bg.jpg";
const Home = () => {
  const skills = [{
    icon: Brain,
    title: "AI/ML & Deep Learning",
    description: "CNNs, LSTMs, Transformers, Computer Vision, NLP, Advanced Neural Networks",
    link: "https://pytorch.org/docs/"
  }, {
    icon: Zap,
    title: "Generative AI & RAG",
    description: "LangChain, LangGraph, CrewAI, AutoGen, Agentic AI, Knowledge Graphs",
    link: "https://docs.langchain.com/"
  }, {
    icon: Code,
    title: "MLOps & LLMOps",
    description: "CI/CD, Model Monitoring, MLflow, Kubeflow, Weights & Biases, Ray",
    link: "https://mlflow.org/docs/"
  }, {
    icon: Cloud,
    title: "Cloud & Infrastructure", 
    description: "AWS, GCP, Azure, Docker, Kubernetes, Airflow, Prefect",
    link: "https://docs.aws.amazon.com/machine-learning/"
  }, {
    icon: Database,
    title: "Data & Databases",
    description: "SQL, NoSQL, Vector DBs (Pinecone, Weaviate, FAISS), Neo4j",
    link: "https://docs.pinecone.io/"
  }, {
    icon: Layers,
    title: "Frameworks & Tools",
    description: "PyTorch, TensorFlow, HuggingFace, OpenAI APIs, FastAPI, Scikit-learn",
    link: "https://huggingface.co/docs"
  }];
  return <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-cover bg-center bg-no-repeat" style={{
        backgroundImage: `url(${heroImage})`
      }}>
          <div className="absolute inset-0 bg-background/40" />
        </div>
        
        <div className="relative z-10 text-center px-4 max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-7xl font-bold mb-6">
            AI Engineer
            <span className="block text-gradient">& ML Expert</span>
          </h1>
          <p className="text-xl md:text-2xl text-foreground/80 mb-8 max-w-2xl mx-auto">AI Engineer | 5+ yrs ML & MLOps | Deep Learning | GenAI | AI Agents | Agentic AI | End-to-End LLMOps</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button asChild size="lg" className="gradient-primary">
              <Link to="/projects">
                View My Work <ArrowRight className="ml-2 h-5 w-5" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link to="/contact">Get In Touch</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Skills Section */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-12">
            Expertise Areas
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {skills.map((skill, index) => (
              <Card key={index} className="p-6 text-center group hover:scale-105 transition-all duration-300 bg-card/50 backdrop-blur-sm border-border/50 cursor-pointer">
                <a
                  href={skill.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block"
                >
                  <div className="w-16 h-16 mx-auto mb-4 rounded-full gradient-primary flex items-center justify-center">
                    <skill.icon className="h-8 w-8 text-primary-foreground" />
                  </div>
                  <h3 className="text-xl font-semibold mb-2">{skill.title}</h3>
                  <p className="text-muted-foreground">{skill.description}</p>
                </a>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 gradient-bg">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Ready to Build Something Amazing?
          </h2>
          <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
            Let's discuss your AI project and turn your ideas into intelligent solutions
          </p>
          <Button asChild size="lg" className="gradient-primary">
            <Link to="/contact">
              Start a Conversation <ArrowRight className="ml-2 h-5 w-5" />
            </Link>
          </Button>
        </div>
      </section>
    </div>;
};
export default Home;