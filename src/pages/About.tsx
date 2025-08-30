import { Calendar, MapPin, Award, Users } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import profileImage from "@/assets/profile-placeholder.jpg";

const About = () => {
  const experience = [
    {
      year: "Nov 2022 – Present",
      role: "Freelance Data Scientist / AI Consultant",
      company: "UpWork",
      description: "Delivered end-to-end ML and GenAI solutions for clients in retail, e-commerce, and finance, enhancing decision-making and boosting revenues.",
      achievements: [
        "Built predictive models for sales forecasting, fraud detection, and customer segmentation using Python (Scikit-learn, TensorFlow, XGBoost), resulting in 10–15% revenue growth",
        "Designed RAG-powered NLP pipelines for social media analytics, increasing sentiment analysis accuracy by 22%",
        "Developed interactive AI dashboards (Streamlit, Power BI) for real-time insights, improving visualization accuracy by 25%"
      ],
      keyProjects: [
        {
          title: "AI-Powered Demand Forecasting for E-Commerce",
          description: "Built a time-series ML pipeline (XGBoost, Prophet) that improved demand prediction accuracy by 18%, optimizing inventory planning and reducing stockouts"
        },
        {
          title: "Customer Segmentation & Personalization in Retail",
          description: "Applied clustering (K-Means, DBSCAN) and RFM analysis to segment customers, enabling targeted campaigns that increased retention by 12%"
        },
        {
          title: "Fraud Detection System for FinTech Client",
          description: "Developed anomaly detection and classification models using Python + Scikit-learn, reducing false positives in fraud detection by 20%"
        },
        {
          title: "NLP-driven Social Media & Review Analytics",
          description: "Designed a RAG-powered sentiment and topic analysis pipeline, improving brand monitoring accuracy by 22% and guiding marketing strategy"
        },
        {
          title: "Interactive AI Dashboards (Streamlit + Power BI)",
          description: "Delivered real-time BI dashboards for retail and finance clients, providing actionable insights that accelerated decision-making by 30%"
        }
      ]
    },
    {
      year: "Oct 2019 – Sep 2022",
      role: "Sr. Representative, Operations",
      company: "Concentrix Daksh Services India Pvt Ltd, Gurugram",
      description: "Developed real-time reporting dashboards and optimized workforce operations through data-driven insights.",
      achievements: [
        "Developed real-time reporting dashboards in Power BI to track key workforce performance metrics, driving a 10% improvement in team productivity",
        "Optimized workforce scheduling by analyzing demand patterns and trends, increasing service level adherence by 18%",
        "Leveraged advanced Excel functions (VLOOKUP, Pivot Tables, formulas) to automate reporting and streamline analytics, reducing manual work by 2 hours per week and improving workflow efficiency by 20%",
        "Provided actionable insights to support data-driven decision-making across operations teams"
      ]
    },
    {
      year: "Feb 2017 – Mar 2019",
      role: "Data Analyst, Content Engineering",
      company: "GlobalLogic Technologies Limited, Gurugram",
      description: "Conducted large-scale data analysis and built automated BI dashboards to improve operational efficiency.",
      achievements: [
        "Conducted large-scale data analysis using SQL and Python, improving operational efficiency by 15%",
        "Built and automated BI dashboards (Power BI + DAX + SQL) for data accuracy and KPIs tracking",
        "Spearheaded DAX-driven advanced analytics improving analysis efficiency by 50%",
        "Collaborated with engineers on data workflows that enabled integration of predictive models in later phases"
      ]
    },
  ];

  const skills = [
    { name: "Python", url: "https://www.python.org/" },
    { name: "TensorFlow", url: "https://www.tensorflow.org/" },
    { name: "PyTorch", url: "https://pytorch.org/" },
    { name: "Kubernetes", url: "https://kubernetes.io/" },
    { name: "Docker", url: "https://www.docker.com/" },
    { name: "MLflow", url: "https://mlflow.org/" },
    { name: "Airflow", url: "https://airflow.apache.org/" },
    { name: "AWS", url: "https://aws.amazon.com/" },
    { name: "GCP", url: "https://cloud.google.com/" },
    { name: "Azure", url: "https://azure.microsoft.com/" },
    { name: "Transformers", url: "https://huggingface.co/transformers/" },
    { name: "LangChain", url: "https://python.langchain.com/" },
    { name: "Vector Databases", url: "https://www.pinecone.io/" },
    { name: "CI/CD", url: "https://github.com/features/actions" },
    { name: "Monitoring", url: "https://prometheus.io/" },
    { name: "A/B Testing", url: "https://www.optimizely.com/" }
  ];

  const achievements = [
    { icon: Award, title: "15+ ML Models", description: "Deployed to production" },
    { icon: Users, title: "50+ Projects", description: "Successfully delivered" },
    { icon: Calendar, title: "5+ Years", description: "ML & AI Experience" },
  ];

  return (
    <div className="min-h-screen pt-20 px-4">
      <div className="container mx-auto">
        {/* Hero Section */}
        <section className="py-20 text-center">
          <div className="w-48 h-48 mx-auto mb-8 rounded-full overflow-hidden">
            <img
              src={profileImage}
              alt="AI Engineer Profile"
              className="w-full h-full object-cover"
            />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">About Me</h1>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            AI Engineer with 5+ years of experience delivering Machine Learning, Deep Learning, Generative AI, and Agentic AI solutions.
            
            Proven expertise in MLOps and LLMOps for end-to-end ML/LLM pipelines including data ingestion, model training, RAG, deployment, monitoring, and governance. 
            Skilled at building multi-agent systems, predictive pipelines, and domain-specific GenAI applications. 
            
            Strong track record of reducing costs, improving accuracy, and scaling AI systems in production environments.
          </p>
        </section>

        {/* Achievements */}
        <section className="py-16">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {achievements.map((achievement, index) => (
              <Card key={index} className="p-6 text-center bg-card/50 backdrop-blur-sm">
                <achievement.icon className="h-8 w-8 mx-auto mb-4 text-primary" />
                <h3 className="text-2xl font-bold mb-2">{achievement.title}</h3>
                <p className="text-muted-foreground text-sm">{achievement.description}</p>
              </Card>
            ))}
          </div>
        </section>

        <section className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12">Professional Journey</h2>
          <div className="space-y-8">
            {experience.map((exp, index) => (
              <Card key={index} className="p-6 bg-card/50 backdrop-blur-sm">
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col md:flex-row md:items-start gap-4">
                    <Badge variant="outline" className="w-fit text-primary border-primary">
                      {exp.year}
                    </Badge>
                    <div className="flex-1">
                      <h3 className="text-xl font-bold mb-1">{exp.role}</h3>
                      <p className="text-primary font-semibold mb-3">{exp.company}</p>
                      <p className="text-muted-foreground mb-4">{exp.description}</p>
                      
                      {exp.achievements && (
                        <div className="mb-4">
                          <ul className="space-y-2">
                            {exp.achievements.map((achievement, achIndex) => (
                              <li key={achIndex} className="flex items-start gap-2">
                                <span className="text-primary mt-2">•</span>
                                <span className="text-sm text-muted-foreground">{achievement}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      
                      {exp.keyProjects && (
                        <div>
                          <h4 className="text-lg font-semibold mb-3 text-primary">Key Projects:</h4>
                          <div className="space-y-3">
                            {exp.keyProjects.map((project, projIndex) => (
                              <div key={projIndex} className="border-l-2 border-primary/30 pl-4">
                                <h5 className="font-semibold mb-1">{project.title}</h5>
                                <p className="text-sm text-muted-foreground">{project.description}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Skills Showcase */}
        <section className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12">Skills Showcase</h2>
          
          {/* Core AI Skills */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">🧠</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Generative AI & Agentic AI</h3>
                  <p className="text-sm text-muted-foreground">Multi-Agent Systems | Domain-Specific GenAI | AI Automation | Knowledge Graphs</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">🔮</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">LLMOps</h3>
                  <p className="text-sm text-muted-foreground">RAG | Prompt Engineering | Fine-Tuning | LLM Deployment | Monitoring | Cost Optimization</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">🤖</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Machine Learning</h3>
                  <p className="text-sm text-muted-foreground">Regression | Classification | Clustering | Dimensionality Reduction | Anomaly Detection | Predictive Modeling</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">🧬</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Deep Learning</h3>
                  <p className="text-sm text-muted-foreground">MLP | CNNs | RNNs | LSTM | GRU | Transformers | GANs | Diffusion Models</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">🗣️</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Natural Language Processing (NLP)</h3>
                  <p className="text-sm text-muted-foreground">Text Classification | Sentiment Analysis | NER | Summarization | Conversational AI | RAG</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">⚙️</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">MLOps & Deployment</h3>
                  <p className="text-sm text-muted-foreground">Pipelines | CI/CD GitHub Actions 🚀 | Model Monitoring | Drift Detection | GitHub 🐙 | Docker 🐳 | MLflow 📈 | DVC 📂 | DagsHub 🌟 | AWS ☁️ | GCP 🌐 | Azure 🔷</p>
                </div>
              </div>
            </Card>
          </div>

          {/* Data & Analysis Skills */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">📊</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Data Analysis & Visualization</h3>
                  <p className="text-sm text-muted-foreground">SQL | Excel | Power BI | Pandas | NumPy | Matplotlib | Seaborn | SciPy | KPI Reporting</p>
                </div>
              </div>
            </Card>

            <Card className="p-6 bg-gradient-to-br from-card/80 to-muted/50 backdrop-blur-sm hover:shadow-lg transition-all duration-300">
              <div className="flex items-start gap-3 mb-3">
                <span className="text-2xl">📝</span>
                <div>
                  <h3 className="text-lg font-semibold text-primary">Data Formats</h3>
                  <p className="text-sm text-muted-foreground">Markdown | HTML | XML | JSON</p>
                </div>
              </div>
            </Card>
          </div>

          {/* Frameworks & Tools */}
          <div className="mt-12">
            <h3 className="text-2xl font-bold text-center mb-8">Frameworks & Tools</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">⚡</span>
                  <h4 className="font-semibold text-primary">ML Libraries</h4>
                </div>
                <p className="text-sm text-muted-foreground">Scikit-learn | XGBoost | LightGBM | CatBoost</p>
              </Card>

              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🔶</span>
                  <h4 className="font-semibold text-primary">DL Frameworks</h4>
                </div>
                <p className="text-sm text-muted-foreground">TensorFlow | PyTorch 🔥 | Keras</p>
              </Card>

              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">📚</span>
                  <h4 className="font-semibold text-primary">NLP Frameworks</h4>
                </div>
                <p className="text-sm text-muted-foreground">Hugging Face 🤗 | SpaCy | NLTK</p>
              </Card>

              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🔗</span>
                  <h4 className="font-semibold text-primary">AI Agent Frameworks</h4>
                </div>
                <p className="text-sm text-muted-foreground">LangChain | LangGraph | LangFlow | n8n</p>
              </Card>

              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🖥️</span>
                  <h4 className="font-semibold text-primary">Backend</h4>
                </div>
                <p className="text-sm text-muted-foreground">Flask | FastAPI ⚡ | Django</p>
              </Card>

              <Card className="p-4 bg-gradient-to-r from-card/60 to-muted/40 backdrop-blur-sm hover:shadow-md transition-all duration-300">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-lg">🎨</span>
                  <h4 className="font-semibold text-primary">Frontend</h4>
                </div>
                <p className="text-sm text-muted-foreground">Streamlit 📊 | HTML | CSS</p>
              </Card>
            </div>
          </div>
        </section>

        {/* Philosophy */}
        <section className="py-16">
          <Card className="p-8 bg-gradient-to-r from-card/50 to-muted/30 backdrop-blur-sm">
            <h2 className="text-3xl font-bold mb-6 text-center">My Philosophy</h2>
            <p className="text-lg text-muted-foreground text-center max-w-3xl mx-auto leading-relaxed">
              "I believe AI should augment human capabilities, not replace them. My approach focuses on 
              building robust, ethical, and scalable AI systems that solve real-world problems while 
              maintaining transparency and reliability."
            </p>
          </Card>
        </section>
      </div>
    </div>
  );
};

export default About;
