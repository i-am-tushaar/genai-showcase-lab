import { ArrowRight, Brain, Code, Zap, Cloud, Database, Layers, Download } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import heroImage from "@/assets/hero-bg.jpg";
const Home = () => {
  const skills = [{
    icon: Zap,
    title: "🤝 Generative AI & Agentic AI",
    description: "LangChain, LangGraph, LangFlow, Multi-Agent Systems, RAG, Knowledge Graphs, AI Automation",
    link: "https://docs.langchain.com/"
  }, {
    icon: Code,
    title: "🔮 LLMOps & Advanced MLOps",
    description: "End-to-End LLM Pipelines | Prompt Engineering | CI/CD (GitHub Actions 🚀) | MLflow 📈 | DVC 📂 | Docker 🐳 | DagsHub 🌟 | Model Monitoring & Drift Detection",
    link: "https://mlflow.org/"
  }, {
    icon: Brain,
    title: "🧠 AI/ML & Deep Learning",
    description: "Regression, Classification, Clustering | CNNs, RNNs, LSTM, Transformers | GANs, Diffusion Models | Computer Vision, Predictive Modeling",
    link: "https://pytorch.org/docs/"
  }, {
    icon: Brain,
    title: "🗣️ Natural Language Processing (NLP)",
    description: "Text Classification, Sentiment Analysis, Summarization, Conversational AI, NER, RAG-powered NLP Pipelines",
    link: "https://huggingface.co/docs"
  }, {
    icon: Cloud,
    title: "☁️ Cloud & Infrastructure", 
    description: "AWS ☁️ | GCP 🌐 | Azure 🔷 | Docker 🐳 | Kubernetes | Airflow | Prefect | GitHub 🐙 CI/CD",
    link: "https://docs.aws.amazon.com/machine-learning/"
  }, {
    icon: Database,
    title: "📊 Data & Analytics",
    description: "SQL | Pandas | NumPy | Matplotlib | Seaborn | Power BI | Streamlit 📊 | KPI Reporting | Data Cleaning & EDA",
    link: "https://pandas.pydata.org/docs/"
  }, {
    icon: Layers,
    title: "🖥️ Frameworks & Tools",
    description: "PyTorch 🔥 | TensorFlow 🔶 | Hugging Face 🤗 | Scikit-learn | FastAPI ⚡ | Flask | Django | HTML, CSS, Markdown, JSON, XML",
    link: "https://docs.djangoproject.com/"
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
            <span className="block text-gradient">AI/ML Engineer</span>
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
            <Button asChild variant="secondary" size="lg">
              <a 
                href="https://your-resume-download-link.com" 
                target="_blank" 
                rel="noopener noreferrer"
              >
                Download Resume <Download className="ml-2 h-5 w-5" />
              </a>
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