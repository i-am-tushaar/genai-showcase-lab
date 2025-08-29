import { Calendar, MapPin, Award, Users } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import profileImage from "@/assets/profile-placeholder.jpg";

const About = () => {
  const experience = [
    {
      year: "2019-Present",
      role: "Senior AI Engineer",
      company: "Tech Innovation Corp",
      description: "Leading AI initiatives, implementing MLOps pipelines, and developing production-ready ML systems.",
    },
    {
      year: "2017-2019",
      role: "ML Engineer",
      company: "Data Solutions Ltd",
      description: "Built deep learning models for computer vision and NLP applications with 95% accuracy improvements.",
    },
    {
      year: "2015-2017",
      role: "Data Scientist",
      company: "Analytics Startup",
      description: "Developed predictive models and data pipelines for early-stage AI products.",
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
    { icon: MapPin, title: "Global Impact", description: "Across 10+ countries" },
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
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {achievements.map((achievement, index) => (
              <Card key={index} className="p-6 text-center bg-card/50 backdrop-blur-sm">
                <achievement.icon className="h-8 w-8 mx-auto mb-4 text-primary" />
                <h3 className="text-2xl font-bold mb-2">{achievement.title}</h3>
                <p className="text-muted-foreground text-sm">{achievement.description}</p>
              </Card>
            ))}
          </div>
        </section>

        {/* Experience Timeline */}
        <section className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12">Professional Journey</h2>
          <div className="space-y-8">
            {experience.map((exp, index) => (
              <Card key={index} className="p-6 bg-card/50 backdrop-blur-sm">
                <div className="flex flex-col md:flex-row md:items-center gap-4">
                  <Badge variant="outline" className="w-fit text-primary border-primary">
                    {exp.year}
                  </Badge>
                  <div className="flex-1">
                    <h3 className="text-xl font-semibold mb-1">{exp.role}</h3>
                    <p className="text-primary mb-2">{exp.company}</p>
                    <p className="text-muted-foreground">{exp.description}</p>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </section>

        {/* Skills */}
        <section className="py-16">
          <h2 className="text-3xl font-bold text-center mb-12">Technical Skills</h2>
          <div className="flex flex-wrap gap-3 justify-center">
            {skills.map((skill, index) => (
              <a
                key={index}
                href={skill.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-block"
              >
                <Badge
                  variant="secondary"
                  className="px-4 py-2 text-sm hover:bg-primary hover:text-primary-foreground transition-colors cursor-pointer"
                >
                  {skill.name}
                </Badge>
              </a>
            ))}
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
