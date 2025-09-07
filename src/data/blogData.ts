export interface BlogPost {
  id: string;
  title: string;
  description: string;
  content: string;
  date: string;
  tags: string[];
  readTime: string;
  category: string;
  featured?: boolean;
}

export const categories = [
  "Data Analysis",
  "Machine Learning", 
  "Deep Learning",
  "Generative AI",
  "AI Agents",
  "Agentic AI",
  "MCP"
];

export const blogPosts: BlogPost[] = [
  {
    id: "complete-data-analysis-workflow",
    title: "Complete Data Analysis Workflow: From Excel to Power BI (With Python & SQL)",
    description: "Master the end-to-end data analysis process, from initial data exploration in Excel to advanced visualizations in Power BI, with Python and SQL integration.",
    content: `# Complete Data Analysis Workflow: From Excel to Power BI

This comprehensive guide walks you through a complete data analysis workflow, demonstrating how to seamlessly integrate Excel, Python, SQL, and Power BI to create powerful analytical solutions.

## Phase 1: Initial Data Exploration with Excel

### Data Assessment and Cleaning
Excel remains one of the most accessible tools for initial data exploration and quick analysis.

**Excel Formulas for Data Quality Assessment:**
\`\`\`excel
// Check for duplicates
=COUNTIF(A:A, A2) > 1

// Identify missing values
=IF(ISBLANK(A2), "Missing", "Present")

// Data type validation
=IF(ISNUMBER(A2), "Number", IF(ISTEXT(A2), "Text", "Other"))

// Outlier detection using IQR
=IF(OR(A2<QUARTILE($A$2:$A$1000,1)-1.5*(QUARTILE($A$2:$A$1000,3)-QUARTILE($A$2:$A$1000,1)),
       A2>QUARTILE($A$2:$A$1000,3)+1.5*(QUARTILE($A$2:$A$1000,3)-QUARTILE($A$2:$A$1000,1))),
   "Outlier", "Normal")
\`\`\`

### Pivot Tables for Initial Analysis
\`\`\`excel
// Create dynamic pivot table summaries
// 1. Insert > PivotTable
// 2. Drag fields to appropriate areas:
//    - Rows: Categories/Dimensions
//    - Values: Metrics (Sum, Average, Count)
//    - Filters: Date ranges, segments
\`\`\`

### Advanced Excel Functions for Analysis
\`\`\`excel
// Dynamic arrays for data analysis (Excel 365)
=FILTER(A2:E1000, (C2:C1000>100)*(D2:D1000="Active"))

// Statistical analysis
=CORREL(A2:A1000, B2:B1000)  // Correlation coefficient
=SLOPE(A2:A1000, B2:B1000)   // Regression slope
=RSQ(A2:A1000, B2:B1000)     // R-squared value

// Conditional aggregations
=SUMIFS(Sales, Region, "North", Date, ">="&DATE(2024,1,1))
=AVERAGEIFS(Performance, Category, "A", Status, "Complete")
\`\`\`

## Phase 2: Advanced Analysis with Python

### Data Import and Integration
\`\`\`python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import plotly.express as px
import plotly.graph_objects as go

# Read Excel data with multiple sheets
excel_data = pd.read_excel('analysis_data.xlsx', sheet_name=None)
df_main = excel_data['MainData']
df_lookup = excel_data['LookupTable']

# Connect to SQL database for additional data
engine = create_engine('postgresql://user:password@localhost:5432/database')
sql_data = pd.read_sql_query("""
    SELECT customer_id, transaction_date, amount, product_category
    FROM transactions 
    WHERE transaction_date >= '2024-01-01'
""", engine)

# Merge Excel and SQL data
combined_data = df_main.merge(sql_data, on='customer_id', how='left')
\`\`\`

### Statistical Analysis and Feature Engineering
\`\`\`python
# Advanced statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Descriptive statistics with confidence intervals
def calculate_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    h = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - h, mean + h

# Feature engineering
df['month'] = pd.to_datetime(df['date']).dt.month
df['quarter'] = pd.to_datetime(df['date']).dt.quarter
df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek

# Create rolling metrics
df['rolling_avg_30d'] = df.groupby('customer_id')['amount'].rolling(window=30).mean().reset_index(drop=True)
df['cumulative_total'] = df.groupby('customer_id')['amount'].cumsum()

# Customer segmentation using RFM analysis
def calculate_rfm(df):
    current_date = df['date'].max()
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (current_date - x.max()).days,  # Recency
        'transaction_id': 'count',                        # Frequency
        'amount': 'sum'                                   # Monetary
    })
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # Create RFM scores
    rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1,2,3,4,5])
    
    return rfm

rfm_analysis = calculate_rfm(combined_data)
\`\`\`

### Advanced Visualizations
\`\`\`python
# Create comprehensive analysis dashboard
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Time series analysis
axes[0,0].plot(df.groupby('date')['amount'].sum())
axes[0,0].set_title('Revenue Trend Over Time')
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Distribution analysis
sns.histplot(df['amount'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Transaction Amount Distribution')

# 3. Correlation heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0,2])
axes[0,2].set_title('Feature Correlations')

# 4. Category performance
category_performance = df.groupby('category')['amount'].agg(['sum', 'mean', 'count'])
category_performance.plot(kind='bar', ax=axes[1,0])
axes[1,0].set_title('Performance by Category')

# 5. Customer segmentation
rfm_analysis.plot.scatter(x='frequency', y='monetary', c='recency', 
                         colormap='viridis', ax=axes[1,1])
axes[1,1].set_title('Customer Segmentation (RFM)')

# 6. Outlier analysis
sns.boxplot(data=df, x='category', y='amount', ax=axes[1,2])
axes[1,2].set_title('Outlier Analysis by Category')
axes[1,2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
\`\`\`

## Phase 3: Database Integration with SQL

### Advanced SQL Queries for Analysis
\`\`\`sql
-- Customer Lifetime Value Analysis
WITH customer_metrics AS (
    SELECT 
        customer_id,
        MIN(transaction_date) as first_purchase,
        MAX(transaction_date) as last_purchase,
        COUNT(*) as transaction_count,
        SUM(amount) as total_spent,
        AVG(amount) as avg_order_value,
        EXTRACT(DAYS FROM MAX(transaction_date) - MIN(transaction_date)) as customer_lifespan
    FROM transactions
    GROUP BY customer_id
),
clv_calculation AS (
    SELECT *,
        CASE 
            WHEN customer_lifespan > 0 
            THEN (total_spent / customer_lifespan) * 365
            ELSE total_spent
        END as annual_clv
    FROM customer_metrics
)
SELECT 
    quartiles.quartile,
    COUNT(*) as customer_count,
    AVG(annual_clv) as avg_clv,
    MIN(annual_clv) as min_clv,
    MAX(annual_clv) as max_clv
FROM (
    SELECT *,
        NTILE(4) OVER (ORDER BY annual_clv) as quartile
    FROM clv_calculation
) quartiles
GROUP BY quartiles.quartile
ORDER BY quartiles.quartile;

-- Advanced Time Series Analysis
WITH daily_metrics AS (
    SELECT 
        DATE_TRUNC('day', transaction_date) as date,
        COUNT(*) as transaction_count,
        SUM(amount) as daily_revenue,
        COUNT(DISTINCT customer_id) as unique_customers,
        AVG(amount) as avg_transaction_value
    FROM transactions
    WHERE transaction_date >= CURRENT_DATE - INTERVAL '365 days'
    GROUP BY DATE_TRUNC('day', transaction_date)
),
metrics_with_trends AS (
    SELECT *,
        LAG(daily_revenue) OVER (ORDER BY date) as prev_day_revenue,
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_7day_avg,
        AVG(daily_revenue) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as rolling_30day_avg
    FROM daily_metrics
)
SELECT *,
    CASE 
        WHEN prev_day_revenue IS NOT NULL 
        THEN ((daily_revenue - prev_day_revenue) / prev_day_revenue) * 100
        ELSE 0
    END as day_over_day_growth
FROM metrics_with_trends
ORDER BY date DESC;

-- Product Performance Analysis with Statistical Measures
SELECT 
    product_category,
    COUNT(*) as total_sales,
    SUM(amount) as total_revenue,
    AVG(amount) as mean_price,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) as median_price,
    STDDEV(amount) as price_std_dev,
    MIN(amount) as min_price,
    MAX(amount) as max_price,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount) as q1,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) as q3
FROM transactions
GROUP BY product_category
ORDER BY total_revenue DESC;
\`\`\`

## Phase 4: Power BI Integration and Advanced Dashboards

### Data Model Setup
\`\`\`dax
// Create calculated columns for enhanced analysis

// Customer Segment (DAX)
CustomerSegment = 
SWITCH(TRUE(),
    Customers[TotalSpent] >= 10000, "VIP",
    Customers[TotalSpent] >= 5000, "Premium",
    Customers[TotalSpent] >= 1000, "Standard",
    "Basic"
)

// Revenue Growth Rate
RevenueGrowthRate = 
VAR CurrentMonthRevenue = SUM(Transactions[Amount])
VAR PreviousMonthRevenue = 
    CALCULATE(
        SUM(Transactions[Amount]),
        DATEADD(Transactions[Date], -1, MONTH)
    )
RETURN
    DIVIDE(
        CurrentMonthRevenue - PreviousMonthRevenue,
        PreviousMonthRevenue
    )

// Customer Retention Rate
CustomerRetentionRate = 
VAR CustomersThisMonth = DISTINCTCOUNT(Transactions[CustomerID])
VAR CustomersLastMonth = 
    CALCULATE(
        DISTINCTCOUNT(Transactions[CustomerID]),
        DATEADD(Transactions[Date], -1, MONTH)
    )
VAR ReturningCustomers = 
    CALCULATE(
        DISTINCTCOUNT(Transactions[CustomerID]),
        FILTER(
            Transactions,
            Transactions[CustomerID] IN 
                CALCULATETABLE(
                    VALUES(Transactions[CustomerID]),
                    DATEADD(Transactions[Date], -1, MONTH)
                )
        )
    )
RETURN
    DIVIDE(ReturningCustomers, CustomersLastMonth)
\`\`\`

### Advanced Measures for Business Intelligence
\`\`\`dax
// Year-over-Year Growth
YoY_Growth = 
VAR CurrentYearValue = SUM(Transactions[Amount])
VAR PreviousYearValue = 
    CALCULATE(
        SUM(Transactions[Amount]),
        SAMEPERIODLASTYEAR(Transactions[Date])
    )
RETURN
    DIVIDE(
        CurrentYearValue - PreviousYearValue,
        PreviousYearValue
    )

// Rolling 12-Month Average
Rolling12MonthAvg = 
AVERAGEX(
    DATESINPERIOD(
        Transactions[Date],
        MAX(Transactions[Date]),
        -12,
        MONTH
    ),
    [MonthlyRevenue]
)

// Customer Lifetime Value Prediction
PredictedCLV = 
VAR AvgOrderValue = AVERAGE(Transactions[Amount])
VAR PurchaseFrequency = DIVIDE(COUNT(Transactions[TransactionID]), DISTINCTCOUNT(Transactions[CustomerID]))
VAR CustomerLifespan = 24 // months - adjust based on business
RETURN
    AvgOrderValue * PurchaseFrequency * CustomerLifespan
\`\`\`

## Integration Workflow Summary

### Data Flow Architecture
1. **Excel** → Initial data collection and basic analysis
2. **Python** → Advanced statistical analysis and feature engineering  
3. **SQL Database** → Scalable data storage and complex queries
4. **Power BI** → Interactive dashboards and business intelligence

### Best Practices for Workflow Integration
- **Version Control**: Use Git for Python scripts and SQL queries
- **Data Validation**: Implement checks at each stage of the pipeline
- **Documentation**: Maintain clear documentation for all transformations
- **Automation**: Schedule regular data updates using tools like Apache Airflow
- **Testing**: Validate results across all platforms for consistency

### Performance Optimization Tips
- **Excel**: Use structured tables and avoid volatile functions
- **Python**: Leverage vectorized operations with pandas and numpy
- **SQL**: Optimize queries with proper indexing and query planning
- **Power BI**: Use DirectQuery for large datasets, Import for smaller ones

This comprehensive workflow ensures robust, scalable data analysis that can grow with your organization's needs.`,
    date: "2024-03-20",
    tags: ["Excel", "Python", "SQL", "Power BI", "Data Analysis", "Workflow"],
    readTime: "25 min read",
    category: "Data Analysis",
    featured: true
  },
  {
    id: "feature-engineering-masterclass",
    title: "Feature Engineering Masterclass: From Raw Data to ML-Ready Features",
    description: "Comprehensive guide to feature engineering techniques including encoding, scaling, selection, and creation strategies for improved model performance.",
    content: `# Feature Engineering Masterclass

Feature engineering is often the difference between a mediocre model and a breakthrough solution. It's the art and science of transforming raw data into features that effectively represent the underlying problem.

## The Feature Engineering Pipeline

### 1. Handling Categorical Variables
\`\`\`python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder, BinaryEncoder

# One-hot encoding for low cardinality
encoder = OneHotEncoder(sparse=False, drop='first')
encoded_features = encoder.fit_transform(df[['category']])

# Target encoding for high cardinality
target_encoder = TargetEncoder()
df['category_encoded'] = target_encoder.fit_transform(df['category'], df['target'])
\`\`\`

### 2. Numerical Feature Scaling
\`\`\`python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard scaling (z-score normalization)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numerical_columns])

# Robust scaling (less sensitive to outliers)
robust_scaler = RobustScaler()
df_robust = robust_scaler.fit_transform(df[numerical_columns])
\`\`\`

### 3. Feature Creation Techniques
\`\`\`python
# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df[['feature1', 'feature2']])

# Domain-specific features
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100], labels=['young', 'adult', 'middle', 'senior'])
df['total_spend_per_transaction'] = df['total_amount'] / df['transaction_count']
\`\`\`

### 4. Time-Based Features
\`\`\`python
# Extract datetime components
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])

# Lag features for time series
df['sales_lag_1'] = df['sales'].shift(1)
df['sales_rolling_mean_7d'] = df['sales'].rolling(window=7).mean()
\`\`\`

### 5. Feature Selection Methods
\`\`\`python
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier

# Statistical feature selection
selector = SelectKBest(score_func=chi2, k=10)
selected_features = selector.fit_transform(X, y)

# Recursive Feature Elimination
rf = RandomForestClassifier()
rfe = RFE(estimator=rf, n_features_to_select=10)
selected_features = rfe.fit_transform(X, y)
\`\`\`

## Advanced Techniques
- **Binning**: Converting continuous variables to categorical
- **Interaction Terms**: Capturing feature relationships  
- **Dimensionality Reduction**: PCA, t-SNE for high-dimensional data
- **Feature Crosses**: Creating combinations of categorical features

## Validation Strategies
Always validate feature engineering choices:
1. Cross-validation performance improvement
2. Feature importance analysis
3. Business logic validation
4. Computational cost assessment

Remember: Good features make simple models work well, while poor features make even sophisticated models struggle.`,
    date: "2024-02-28",
    tags: ["Machine Learning", "Feature Engineering", "Data Preprocessing"],
    readTime: "15 min read",
    category: "Machine Learning"
  },
  {
    id: "deep-learning-optimization",
    title: "Deep Learning Model Optimization: From Training to Deployment",
    description: "Advanced techniques for optimizing deep learning models including hyperparameter tuning, regularization, and efficient inference strategies.",
    content: `# Deep Learning Model Optimization

Optimizing deep learning models requires a systematic approach spanning architecture design, training strategies, and deployment considerations.

## Training Optimization

### 1. Learning Rate Strategies
\`\`\`python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Cyclical learning rates
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Adaptive learning rate reduction
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
\`\`\`

### 2. Regularization Techniques
\`\`\`python
# Dropout
nn.Dropout(p=0.5)

# Batch normalization
nn.BatchNorm2d(num_features)

# Weight decay (L2 regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
\`\`\`

### 3. Advanced Optimization
- **Gradient Clipping**: Preventing exploding gradients
- **Mixed Precision Training**: FP16 for faster training
- **Gradient Accumulation**: Simulating larger batch sizes

## Model Architecture Optimization
- **Skip Connections**: ResNet-style connections
- **Attention Mechanisms**: Focusing on relevant features  
- **Efficient Architectures**: MobileNets, EfficientNets

## Deployment Optimization
- **Model Quantization**: Reducing model size
- **Knowledge Distillation**: Teacher-student training
- **ONNX Conversion**: Cross-platform deployment

The key is balancing model performance with computational constraints for your specific use case.`,
    date: "2024-02-25",
    tags: ["Deep Learning", "Optimization", "PyTorch"],
    readTime: "18 min read",
    category: "Deep Learning",
    featured: true
  },
  {
    id: "generative-ai-applications",
    title: "Building Production-Ready Generative AI Applications",
    description: "Complete guide to developing and deploying generative AI applications including prompt engineering, fine-tuning, and scaling strategies.",
    content: `# Building Production-Ready Generative AI Applications

Generative AI has transformed how we build intelligent applications. This guide covers the entire lifecycle from prototype to production-ready systems.

## Application Architecture

### 1. Core Components
\`\`\`python
class GenerativeAIApp:
    def __init__(self):
        self.prompt_template = PromptTemplate()
        self.model_client = OpenAIClient()
        self.vector_store = PineconeIndex()
        self.cache_layer = RedisCache()
    
    def generate_response(self, user_input):
        # Retrieve relevant context
        context = self.vector_store.search(user_input)
        
        # Format prompt with context
        prompt = self.prompt_template.format(
            context=context,
            user_input=user_input
        )
        
        # Generate response
        response = self.model_client.generate(prompt)
        return response
\`\`\`

### 2. Prompt Engineering Strategies
- **Chain-of-Thought**: Step-by-step reasoning
- **Few-Shot Learning**: Providing examples  
- **Role-Based Prompts**: Defining AI persona
- **Template Versioning**: Managing prompt iterations

### 3. Quality Control
\`\`\`python
class OutputValidator:
    def __init__(self):
        self.safety_filter = SafetyFilter()
        self.factual_checker = FactualChecker()
    
    def validate_output(self, response):
        if not self.safety_filter.is_safe(response):
            return self.generate_fallback_response()
        
        confidence = self.factual_checker.check_facts(response)
        if confidence < 0.8:
            response += "\n\n*Please verify this information independently."
        
        return response
\`\`\`

## Production Considerations
- **Cost Optimization**: Token usage monitoring
- **Latency Management**: Response time optimization
- **Scalability**: Load balancing and caching
- **Monitoring**: Quality metrics and user feedback

## Deployment Patterns
- **API Gateway**: Rate limiting and authentication
- **Microservices**: Modular architecture
- **Edge Deployment**: Reducing latency
- **A/B Testing**: Continuous improvement

Success in GenAI requires balancing innovation with operational excellence.`,
    date: "2024-02-20",
    tags: ["Generative AI", "Production AI", "Prompt Engineering"],
    readTime: "14 min read",
    category: "Generative AI"
  },
  {
    id: "autonomous-ai-agents",
    title: "Designing Autonomous AI Agents for Complex Task Execution",
    description: "Architecture patterns and implementation strategies for building AI agents that can autonomously plan, execute, and adapt to achieve complex goals.",
    content: `# Designing Autonomous AI Agents

Autonomous AI agents represent the next evolution in AI systems, capable of independent reasoning, planning, and task execution with minimal human intervention.

## Agent Architecture

### 1. Core Components
\`\`\`python
class AutonomousAgent:
    def __init__(self):
        self.perception = PerceptionModule()
        self.planning = PlanningModule()
        self.execution = ExecutionModule()
        self.memory = MemorySystem()
        self.learning = LearningModule()
    
    def execute_task(self, goal):
        # Perceive current environment
        state = self.perception.observe()
        
        # Plan actions to achieve goal
        plan = self.planning.create_plan(goal, state)
        
        # Execute plan with adaptation
        result = self.execution.execute_with_monitoring(plan)
        
        # Learn from experience
        self.learning.update_from_experience(goal, plan, result)
        
        return result
\`\`\`

### 2. Planning and Reasoning
- **Goal Decomposition**: Breaking complex tasks into subtasks
- **State Space Search**: Finding optimal action sequences
- **Constraint Satisfaction**: Handling real-world limitations
- **Contingency Planning**: Preparing for failure scenarios

### 3. Memory Systems
\`\`\`python
class AgentMemory:
    def __init__(self):
        self.episodic_memory = EpisodicBuffer()  # Past experiences
        self.semantic_memory = KnowledgeGraph()  # Domain knowledge
        self.working_memory = WorkingBuffer()    # Current context
    
    def retrieve_relevant_experience(self, current_situation):
        similar_episodes = self.episodic_memory.find_similar(current_situation)
        return self.extract_patterns(similar_episodes)
\`\`\`

## Autonomous Capabilities
- **Self-Monitoring**: Performance assessment
- **Error Recovery**: Handling unexpected situations  
- **Knowledge Acquisition**: Learning from experience
- **Goal Adaptation**: Adjusting objectives based on context

## Applications
- **Research Assistants**: Literature review and analysis
- **Code Generation**: Autonomous programming
- **Business Process Automation**: End-to-end workflow execution
- **Personal Assistants**: Multi-step task completion

Building effective autonomous agents requires careful balance between autonomy and controllability.`,
    date: "2024-02-15",
    tags: ["AI Agents", "Autonomous Systems", "Planning"],
    readTime: "16 min read",
    category: "AI Agents",
    featured: true
  },
  {
    id: "multi-agent-collaboration",
    title: "Multi-Agent Systems: Orchestrating Collaborative AI Teams",
    description: "Advanced patterns for building systems where multiple AI agents collaborate, communicate, and coordinate to solve complex problems together.",
    content: `# Multi-Agent Systems: Orchestrating Collaborative AI Teams

Multi-agent systems enable complex problem-solving by coordinating multiple specialized AI agents, each contributing unique capabilities to achieve shared objectives.

## System Architecture

### 1. Agent Coordination Patterns
\`\`\`python
class MultiAgentOrchestrator:
    def __init__(self):
        self.agents = {
            'researcher': ResearchAgent(),
            'analyst': AnalysisAgent(), 
            'writer': WritingAgent(),
            'reviewer': ReviewAgent()
        }
        self.communication_bus = MessageBus()
        self.coordinator = CoordinationEngine()
    
    def execute_collaborative_task(self, task):
        # Decompose task into agent-specific subtasks
        subtasks = self.coordinator.decompose_task(task)
        
        # Orchestrate agent collaboration
        results = {}
        for agent_name, subtask in subtasks.items():
            agent = self.agents[agent_name]
            result = agent.execute(subtask)
            results[agent_name] = result
            
            # Share results with other agents
            self.communication_bus.broadcast(agent_name, result)
        
        # Synthesize final result
        return self.coordinator.synthesize_results(results)
\`\`\`

### 2. Communication Protocols
- **Message Passing**: Structured inter-agent communication
- **Shared Memory**: Common knowledge repositories
- **Event-Driven**: Reactive coordination patterns
- **Consensus Mechanisms**: Collaborative decision making

### 3. Specialization Strategies
\`\`\`python
class SpecialistAgent:
    def __init__(self, domain_expertise):
        self.expertise = domain_expertise
        self.tools = self.load_domain_tools()
        self.knowledge_base = self.load_domain_knowledge()
    
    def can_handle_task(self, task):
        return self.expertise.matches(task.domain)
    
    def execute_with_expertise(self, task):
        # Apply domain-specific knowledge and tools
        return self.apply_expertise(task)
\`\`\`

## Collaboration Patterns
- **Pipeline Processing**: Sequential agent processing
- **Parallel Execution**: Concurrent task distribution
- **Hierarchical Coordination**: Manager-worker patterns
- **Peer-to-Peer**: Decentralized collaboration

## Quality Assurance
- **Cross-Validation**: Multiple agents reviewing work
- **Consensus Building**: Agreement on decisions
- **Performance Monitoring**: Individual and team metrics
- **Failure Recovery**: Handling agent failures gracefully

Multi-agent systems excel when individual agent strengths complement each other, creating emergent capabilities greater than the sum of their parts.`,
    date: "2024-02-10",
    tags: ["Agentic AI", "Multi-Agent Systems", "Collaboration"],
    readTime: "13 min read",
    category: "Agentic AI"
  },
  {
    id: "top-data-analysis-techniques",
    title: "Top Data Analysis Techniques Every Analyst Must Master (Excel, SQL, Power BI & Python)",
    description: "Essential data analysis techniques across all major platforms - from Excel formulas to Python libraries, SQL queries to Power BI visualizations.",
    content: `# Top Data Analysis Techniques Every Analyst Must Master

Modern data analysts need proficiency across multiple platforms. This comprehensive guide covers essential techniques in Excel, SQL, Power BI, and Python that every analyst should master.

## Excel: The Foundation of Data Analysis

### 1. Advanced Excel Formulas and Functions

#### Dynamic Array Functions (Excel 365)
\`\`\`excel
// FILTER function for dynamic data extraction
=FILTER(A2:E100, (C2:C100>1000)*(D2:D100="Active"))

// SORT and SORTBY for dynamic sorting
=SORT(A2:C100, 3, -1)  // Sort by 3rd column descending
=SORTBY(A2:B100, C2:C100, -1)  // Sort A2:B100 by values in C2:C100

// UNIQUE for removing duplicates
=UNIQUE(A2:A100)

// XLOOKUP - the modern replacement for VLOOKUP
=XLOOKUP(lookup_value, lookup_array, return_array, if_not_found, match_mode, search_mode)
\`\`\`

#### Statistical Analysis Functions
\`\`\`excel
// Descriptive statistics
=QUARTILE.INC(range, quartile_number)  // Quartiles
=PERCENTILE.INC(range, k)              // Percentiles
=SKEW(range)                           // Skewness
=KURT(range)                           // Kurtosis

// Correlation and regression
=CORREL(array1, array2)                // Correlation coefficient
=SLOPE(known_y_values, known_x_values) // Linear regression slope
=INTERCEPT(known_y_values, known_x_values) // Y-intercept
=RSQ(known_y_values, known_x_values)   // R-squared

// Confidence intervals
=CONFIDENCE.T(alpha, standard_dev, size)
\`\`\`

#### Advanced Conditional Logic
\`\`\`excel
// Nested conditions with multiple criteria
=IFS(condition1, value1, condition2, value2, TRUE, default_value)

// Array-based conditional sums
=SUMPRODUCT((Category="A")*(Status="Active")*Amount)

// Dynamic conditional formatting with formulas
=AND($B2>AVERAGE($B$2:$B$100), $C2>MEDIAN($C$2:$C$100))
\`\`\`

### 2. Pivot Tables and Advanced Analytics

#### Dynamic Pivot Table Techniques
\`\`\`excel
// Create calculated fields in pivot tables
// 1. Click on pivot table
// 2. PivotTable Analyze > Fields, Items & Sets > Calculated Field
// Example: Profit Margin = Sales - Costs

// Grouping dates intelligently
// Right-click on date field in pivot > Group
// Options: Days, Months, Quarters, Years

// Show values as percentage calculations
// Right-click on value field > Show Values As > % of Grand Total
\`\`\`

#### Power Query for Data Transformation
\`\`\`excel
// Data > Get Data > From File/Database/Web
// Transform operations in Power Query Editor:

// Remove duplicates
Table.Distinct(Source)

// Filter rows
Table.SelectRows(Source, each [Column] > 100)

// Add conditional columns
if [Sales] > 1000 then "High" else if [Sales] > 500 then "Medium" else "Low"

// Group and aggregate
Table.Group(Source, {"Category"}, {{"Total Sales", each List.Sum([Sales]), type number}})
\`\`\`

## SQL: Mastering Database Analysis

### 3. Advanced SQL Query Techniques

#### Window Functions for Analytics
\`\`\`sql
-- Ranking and row numbering
SELECT 
    customer_id,
    order_date,
    order_amount,
    ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as order_sequence,
    RANK() OVER (ORDER BY order_amount DESC) as amount_rank,
    DENSE_RANK() OVER (PARTITION BY EXTRACT(YEAR FROM order_date) ORDER BY order_amount DESC) as yearly_rank
FROM orders;

-- Running totals and moving averages
SELECT 
    order_date,
    daily_sales,
    SUM(daily_sales) OVER (ORDER BY order_date ROWS UNBOUNDED PRECEDING) as running_total,
    AVG(daily_sales) OVER (ORDER BY order_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as moving_avg_7day,
    LAG(daily_sales, 1) OVER (ORDER BY order_date) as previous_day,
    LEAD(daily_sales, 1) OVER (ORDER BY order_date) as next_day
FROM daily_sales_summary;

-- Percentile calculations
SELECT 
    product_category,
    price,
    PERCENT_RANK() OVER (PARTITION BY product_category ORDER BY price) as price_percentile,
    NTILE(4) OVER (ORDER BY price) as price_quartile
FROM products;
\`\`\`

#### Complex Analytical Queries
\`\`\`sql
-- Customer Cohort Analysis
WITH cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', MIN(order_date)) as cohort_month
    FROM orders
    GROUP BY customer_id
),
cohort_sizes AS (
    SELECT 
        cohort_month,
        COUNT(*) as cohort_size
    FROM cohorts
    GROUP BY cohort_month
),
cohort_table AS (
    SELECT 
        c.cohort_month,
        DATE_TRUNC('month', o.order_date) as order_month,
        COUNT(DISTINCT o.customer_id) as active_customers
    FROM cohorts c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.cohort_month, DATE_TRUNC('month', o.order_date)
)
SELECT 
    ct.cohort_month,
    ct.order_month,
    ct.active_customers,
    cs.cohort_size,
    ROUND(ct.active_customers::numeric / cs.cohort_size * 100, 2) as retention_rate,
    EXTRACT(MONTH FROM AGE(ct.order_month, ct.cohort_month)) as period_number
FROM cohort_table ct
JOIN cohort_sizes cs ON ct.cohort_month = cs.cohort_month
ORDER BY ct.cohort_month, ct.order_month;

-- Advanced Sales Analytics
WITH sales_metrics AS (
    SELECT 
        DATE_TRUNC('month', order_date) as month,
        customer_id,
        SUM(order_amount) as monthly_spend,
        COUNT(*) as monthly_orders
    FROM orders
    GROUP BY DATE_TRUNC('month', order_date), customer_id
),
customer_segments AS (
    SELECT 
        customer_id,
        AVG(monthly_spend) as avg_monthly_spend,
        AVG(monthly_orders) as avg_monthly_orders,
        CASE 
            WHEN AVG(monthly_spend) > 1000 AND AVG(monthly_orders) > 5 THEN 'High Value Frequent'
            WHEN AVG(monthly_spend) > 1000 THEN 'High Value'
            WHEN AVG(monthly_orders) > 5 THEN 'Frequent'
            ELSE 'Standard'
        END as customer_segment
    FROM sales_metrics
    GROUP BY customer_id
)
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(avg_monthly_spend), 2) as avg_spend,
    ROUND(AVG(avg_monthly_orders), 2) as avg_orders
FROM customer_segments
GROUP BY customer_segment
ORDER BY avg_spend DESC;
\`\`\`

## Python: Advanced Statistical Analysis

### 4. Pandas for Data Manipulation

#### Advanced DataFrame Operations
\`\`\`python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Advanced groupby operations
def calculate_customer_metrics(df):
    return df.groupby('customer_id').agg({
        'order_date': ['min', 'max', 'count'],
        'order_amount': ['sum', 'mean', 'std'],
        'product_id': lambda x: x.nunique()  # Unique products purchased
    }).round(2)

# Multi-level indexing and pivoting
pivot_analysis = df.pivot_table(
    values=['sales', 'profit'],
    index=['year', 'quarter'],
    columns='product_category',
    aggfunc={'sales': 'sum', 'profit': ['sum', 'mean']},
    fill_value=0,
    margins=True
)

# Time series resampling and analysis
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Resample to different frequencies
monthly_summary = df.resample('M').agg({
    'sales': ['sum', 'mean', 'count'],
    'customers': 'nunique'
})

# Rolling calculations
df['sales_ma_7d'] = df['sales'].rolling(window=7).mean()
df['sales_ma_30d'] = df['sales'].rolling(window=30).mean()
df['sales_ewm'] = df['sales'].ewm(span=10).mean()  # Exponential weighted moving average
\`\`\`

#### Statistical Analysis and Hypothesis Testing
\`\`\`python
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

# Comprehensive statistical testing function
def perform_statistical_tests(group1, group2, test_name="Groups"):
    """
    Perform multiple statistical tests comparing two groups
    """
    results = {}
    
    # Descriptive statistics
    results['group1_stats'] = {
        'mean': np.mean(group1),
        'median': np.median(group1),
        'std': np.std(group1),
        'n': len(group1)
    }
    results['group2_stats'] = {
        'mean': np.mean(group2),
        'median': np.median(group2),
        'std': np.std(group2),
        'n': len(group2)
    }
    
    # Normality tests
    _, p_norm1 = stats.shapiro(group1[:5000])  # Shapiro-Wilk (max 5000 samples)
    _, p_norm2 = stats.shapiro(group2[:5000])
    
    results['normality'] = {
        'group1_normal': p_norm1 > 0.05,
        'group2_normal': p_norm2 > 0.05
    }
    
    # Choose appropriate test based on normality
    if results['normality']['group1_normal'] and results['normality']['group2_normal']:
        # Both normal - use t-test
        t_stat, p_value = ttest_ind(group1, group2)
        results['test_used'] = 'Independent t-test'
    else:
        # Non-normal - use Mann-Whitney U test
        u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        results['test_used'] = 'Mann-Whitney U test'
        t_stat = u_stat
    
    results['test_results'] = {
        'statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    return results

# A/B Testing Analysis
def ab_test_analysis(control_group, treatment_group, metric_name):
    """
    Comprehensive A/B test analysis
    """
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(((len(control_group) - 1) * np.var(control_group) + 
                         (len(treatment_group) - 1) * np.var(treatment_group)) / 
                        (len(control_group) + len(treatment_group) - 2))
    
    cohens_d = (np.mean(treatment_group) - np.mean(control_group)) / pooled_std
    
    # Statistical test
    test_results = perform_statistical_tests(control_group, treatment_group)
    
    # Calculate confidence interval for the difference
    diff_mean = np.mean(treatment_group) - np.mean(control_group)
    se_diff = np.sqrt(np.var(control_group)/len(control_group) + 
                     np.var(treatment_group)/len(treatment_group))
    
    ci_lower = diff_mean - 1.96 * se_diff
    ci_upper = diff_mean + 1.96 * se_diff
    
    return {
        'metric': metric_name,
        'control_mean': np.mean(control_group),
        'treatment_mean': np.mean(treatment_group),
        'difference': diff_mean,
        'effect_size_cohens_d': cohens_d,
        'confidence_interval_95': (ci_lower, ci_upper),
        'statistical_test': test_results
    }
\`\`\`

## Power BI: Advanced Visualization and DAX

### 5. DAX Mastery for Business Intelligence

#### Advanced DAX Patterns
\`\`\`dax
// Time Intelligence Functions
YTD_Sales = TOTALYTD(SUM(Sales[Amount]), DateTable[Date])
MTD_Sales = TOTALMTD(SUM(Sales[Amount]), DateTable[Date])
QTD_Sales = TOTALQTD(SUM(Sales[Amount]), DateTable[Date])

// Previous Period Comparisons
Sales_PY = CALCULATE(
    SUM(Sales[Amount]),
    SAMEPERIODLASTYEAR(DateTable[Date])
)

Sales_PM = CALCULATE(
    SUM(Sales[Amount]),
    DATEADD(DateTable[Date], -1, MONTH)
)

// Growth Calculations
YoY_Growth = 
VAR CurrentYear = SUM(Sales[Amount])
VAR PreviousYear = [Sales_PY]
RETURN
    DIVIDE(CurrentYear - PreviousYear, PreviousYear, 0)

// Advanced Filtering with CALCULATE
High_Value_Customers_Sales = 
CALCULATE(
    SUM(Sales[Amount]),
    FILTER(
        Customers,
        Customers[LifetimeValue] > 10000
    ),
    Sales[Status] = "Completed"
)

// Dynamic Segmentation
Customer_Segment = 
VAR CustomerValue = SUM(Sales[Amount])
RETURN
    SWITCH(
        TRUE(),
        CustomerValue >= 10000, "Premium",
        CustomerValue >= 5000, "Gold",
        CustomerValue >= 1000, "Silver",
        "Bronze"
    )
\`\`\`

#### Advanced Measures for Analytics
\`\`\`dax
// Cohort Analysis in DAX
Cohort_Month = 
VAR FirstPurchaseDate = 
    CALCULATE(
        MIN(Sales[Date]),
        ALLEXCEPT(Sales, Sales[CustomerID])
    )
RETURN
    FORMAT(FirstPurchaseDate, "YYYY-MM")

Cohort_Period = 
VAR FirstPurchase = 
    CALCULATE(
        MIN(Sales[Date]),
        ALLEXCEPT(Sales, Sales[CustomerID])
    )
VAR CurrentDate = MIN(Sales[Date])
RETURN
    DATEDIFF(FirstPurchase, CurrentDate, MONTH)

// Customer Lifetime Value Prediction
Predicted_CLV = 
VAR AvgOrderValue = AVERAGE(Sales[Amount])
VAR PurchaseFrequency = 
    DIVIDE(
        COUNT(Sales[SaleID]),
        DISTINCTCOUNT(Sales[CustomerID])
    )
VAR CustomerLifespan = 24 // months
RETURN
    AvgOrderValue * PurchaseFrequency * CustomerLifespan

// Advanced Ranking with Ties
Dense_Rank_Products = 
RANKX(
    ALLSELECTED(Products[ProductName]),
    SUM(Sales[Amount]),
    ,
    DESC,
    DENSE
)
\`\`\`

### 6. Interactive Dashboard Design Principles

#### Advanced Visualization Techniques
- **Conditional Formatting**: Use DAX to create dynamic color schemes
- **Drill-through Pages**: Create detailed views for specific data points
- **Bookmarks and Selection**: Build interactive story-telling experiences
- **Custom Visuals**: Leverage marketplace or create R/Python visuals

#### Performance Optimization Strategies
\`\`\`dax
// Use variables to avoid recalculation
Optimized_Measure = 
VAR TotalSales = SUM(Sales[Amount])
VAR TotalCosts = SUM(Sales[Cost])
VAR Profit = TotalSales - TotalCosts
VAR ProfitMargin = DIVIDE(Profit, TotalSales)
RETURN
    IF(TotalSales > 0, ProfitMargin, BLANK())

// Minimize use of calculated columns, prefer measures
// Use SUMMARIZE instead of ADDCOLUMNS when possible
// Implement proper data modeling with star schema
\`\`\`

## Integration Best Practices

### Cross-Platform Workflow
1. **Excel**: Initial data exploration and quick prototyping
2. **SQL**: Heavy data processing and complex transformations  
3. **Python**: Statistical analysis and machine learning
4. **Power BI**: Interactive dashboards and business reporting

### Data Quality Framework
- **Validation Rules**: Implement at each stage
- **Error Handling**: Graceful degradation strategies
- **Version Control**: Track changes across all platforms
- **Documentation**: Maintain clear process documentation

### Performance Optimization
- **Excel**: Use structured references and avoid volatile functions
- **SQL**: Implement proper indexing and query optimization
- **Python**: Leverage vectorized operations and efficient libraries
- **Power BI**: Optimize data model and use appropriate storage modes

Mastering these techniques across all four platforms will make you a versatile and highly effective data analyst capable of handling any analytical challenge.`,
    date: "2024-03-18",
    tags: ["Excel", "SQL", "Power BI", "Python", "Data Analysis", "Techniques"],
    readTime: "30 min read",
    category: "Data Analysis"
  },
  {
    id: "data-visualization-mastery",
    title: "Data Visualization Mastery: Creating Impactful Charts and Dashboards",
    description: "Master the art of data storytelling through effective visualization techniques, from basic charts to interactive dashboards using Python and modern tools.",
    content: `# Data Visualization Mastery: Creating Impactful Charts and Dashboards

Effective data visualization transforms complex datasets into clear, actionable insights that drive business decisions.

## Visualization Fundamentals

### Choosing the Right Chart Type
\`\`\`python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Bar charts for categorical comparisons
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x='category', y='value', palette='viridis')
plt.title('Category Performance Comparison')
plt.xticks(rotation=45)

# Line charts for time series
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['metric'], linewidth=2, marker='o')
plt.title('Trend Analysis Over Time')
plt.xlabel('Date')
plt.ylabel('Metric Value')
\`\`\`

### Advanced Statistical Visualizations
\`\`\`python
# Correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, fmt='.2f', cbar_kws={'shrink': 0.8})

# Distribution analysis
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Histogram with KDE
sns.histplot(df['variable'], kde=True, ax=axes[0,0])
axes[0,0].set_title('Distribution with Density Curve')

# Box plot for outlier detection
sns.boxplot(data=df, y='variable', ax=axes[0,1])
axes[0,1].set_title('Outlier Detection')

# Q-Q plot for normality
from scipy import stats
stats.probplot(df['variable'], dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot for Normality')

# Violin plot for distribution shape
sns.violinplot(data=df, x='category', y='variable', ax=axes[1,1])
axes[1,1].set_title('Distribution by Category')
\`\`\`

## Interactive Dashboards

### Plotly for Interactive Visualization
\`\`\`python
# Interactive scatter plot with hover information
fig = px.scatter(df, x='feature1', y='feature2', color='category',
                size='size_metric', hover_data=['additional_info'],
                title='Interactive Feature Relationship')

# Add custom hover template
fig.update_traces(
    hovertemplate="<b>%{hovertext}</b><br>" +
                  "Feature 1: %{x}<br>" +
                  "Feature 2: %{y}<br>" +
                  "Category: %{marker.color}<br>" +
                  "<extra></extra>"
)

fig.show()
\`\`\`

### Dashboard Layout with Subplot Framework
\`\`\`python
from plotly.subplots import make_subplots

# Create dashboard with multiple visualizations
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Sales Trend', 'Category Distribution', 
                   'Regional Performance', 'Key Metrics'),
    specs=[[{"secondary_y": True}, {"type": "pie"}],
           [{"type": "bar"}, {"type": "indicator"}]]
)

# Add multiple chart types
fig.add_trace(go.Scatter(x=dates, y=sales, name='Sales'), row=1, col=1)
fig.add_trace(go.Pie(labels=categories, values=counts), row=1, col=2)
fig.add_trace(go.Bar(x=regions, y=performance), row=2, col=1)
fig.add_trace(go.Indicator(mode="number+delta", value=kpi_value), row=2, col=2)
\`\`\`

## Design Principles
- **Clarity**: Remove unnecessary elements and focus on the message
- **Consistency**: Use consistent colors, fonts, and styling
- **Context**: Provide appropriate scales, labels, and annotations
- **Color Theory**: Use color purposefully to highlight insights

## Best Practices
- Start with the question you want to answer
- Choose appropriate chart types for your data
- Optimize for your audience and medium
- Test with real users for feedback
- Iterate and refine based on usage patterns

Great visualizations don't just show data—they reveal insights and inspire action.`,
    date: "2024-03-10",
    tags: ["Visualization", "Python", "Plotly", "Dashboard Design"],
    readTime: "16 min read",
    category: "Data Analysis"
  },
  {
    id: "mlops-production-pipeline",
    title: "MLOps: Building Production-Ready Machine Learning Pipelines",
    description: "Complete guide to implementing MLOps practices including CI/CD for ML, model monitoring, automated retraining, and production deployment strategies.",
    content: `# MLOps: Building Production-Ready Machine Learning Pipelines

MLOps bridges the gap between machine learning development and production deployment, ensuring reliable, scalable, and maintainable ML systems.

## MLOps Architecture Overview

### Core Components
\`\`\`python
# MLOps Pipeline Structure
class MLOpsPipeline:
    def __init__(self):
        self.data_validator = DataValidator()
        self.feature_store = FeatureStore()
        self.model_registry = ModelRegistry()
        self.monitoring = ModelMonitoring()
        self.deployment = ModelDeployment()
    
    def run_training_pipeline(self):
        # Data validation and quality checks
        validated_data = self.data_validator.validate(raw_data)
        
        # Feature engineering and storage
        features = self.feature_store.create_features(validated_data)
        
        # Model training with experiment tracking
        model = self.train_model_with_tracking(features)
        
        # Model validation and registration
        if self.validate_model(model):
            self.model_registry.register(model)
        
        return model
\`\`\`

### Data Pipeline Management
\`\`\`python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Scalable data processing pipeline
def create_data_pipeline():
    with beam.Pipeline(options=PipelineOptions()) as pipeline:
        (pipeline
         | 'Read Data' >> beam.io.ReadFromBigQuery(query=data_query)
         | 'Validate Data' >> beam.Map(validate_data_quality)
         | 'Feature Engineering' >> beam.Map(engineer_features)
         | 'Write Features' >> beam.io.WriteToBigQuery(feature_table)
        )

# Data quality monitoring
class DataQualityMonitor:
    def __init__(self):
        self.quality_checks = [
            self.check_missing_values,
            self.check_data_drift,
            self.check_schema_compliance
        ]
    
    def validate_batch(self, data_batch):
        results = {}
        for check in self.quality_checks:
            results[check.__name__] = check(data_batch)
        return results
\`\`\`

## Model Training and Experimentation

### Experiment Tracking with MLflow
\`\`\`python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train_model_with_tracking(X_train, y_train, hyperparameters):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(hyperparameters)
        
        # Train model
        model = RandomForestClassifier(**hyperparameters)
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        
        mlflow.log_metrics({
            'accuracy': accuracy,
            'precision': precision,
            'training_data_size': len(X_train)
        })
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model
\`\`\`

### Automated Model Validation
\`\`\`python
class ModelValidator:
    def __init__(self, baseline_metrics):
        self.baseline_metrics = baseline_metrics
        self.validation_tests = [
            self.test_model_performance,
            self.test_prediction_bias,
            self.test_model_stability
        ]
    
    def validate_model(self, model, test_data):
        validation_results = {}
        
        for test in self.validation_tests:
            result = test(model, test_data)
            validation_results[test.__name__] = result
        
        # Overall validation decision
        passed_validation = all(
            result['passed'] for result in validation_results.values()
        )
        
        return passed_validation, validation_results
\`\`\`

## Production Deployment

### Model Serving with FastAPI
\`\`\`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Validate input
        features = np.array(request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            model_version="v1.2.3"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
\`\`\`

### Container Deployment
\`\`\`dockerfile
# Dockerfile for model serving
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
\`\`\`

## Monitoring and Maintenance

### Model Drift Detection
\`\`\`python
from scipy import stats
import numpy as np

class DriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        
    def detect_data_drift(self, new_data, threshold=0.05):
        drift_results = {}
        
        for column in self.reference_data.columns:
            # Kolmogorov-Smirnov test for distribution drift
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data[column],
                new_data[column]
            )
            
            drift_results[column] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': p_value < threshold
            }
        
        return drift_results
\`\`\`

## CI/CD for ML
- **Continuous Integration**: Automated testing of data and models
- **Continuous Deployment**: Automated model deployment pipelines
- **Model Versioning**: Track model lineage and reproducibility
- **Rollback Strategies**: Safe deployment with quick rollback capabilities

MLOps transforms ML from experimental prototypes to reliable production systems that deliver consistent business value.`,
    date: "2024-02-22",
    tags: ["MLOps", "Production ML", "CI/CD", "Model Deployment"],
    readTime: "20 min read",
    category: "Machine Learning"
  },
  {
    id: "model-interpretability-explainable-ai",
    title: "Model Interpretability and Explainable AI: Making Black Boxes Transparent",
    description: "Comprehensive guide to understanding and explaining machine learning models using SHAP, LIME, and other interpretability techniques for trustworthy AI.",
    content: `# Model Interpretability and Explainable AI

Model interpretability is crucial for building trust, ensuring fairness, and meeting regulatory requirements in machine learning applications.

## Why Interpretability Matters

### Business and Ethical Requirements
- **Trust and Adoption**: Users need to understand AI decisions
- **Regulatory Compliance**: GDPR, financial regulations require explainability
- **Debugging**: Identifying model failures and biases
- **Feature Validation**: Ensuring models use relevant features

## Global Interpretability Methods

### Feature Importance Analysis
\`\`\`python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Train model and analyze feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Built-in feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

# Permutation importance (more reliable)
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42
)

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.title('Feature Importance Analysis')
\`\`\`

### Partial Dependence Plots
\`\`\`python
from sklearn.inspection import partial_dependence, plot_partial_dependence

# Analyze how individual features affect predictions
features_to_plot = [0, 1, (0, 1)]  # Individual features and interactions
fig, ax = plt.subplots(figsize=(12, 4))

plot_partial_dependence(
    rf_model, X_train, features_to_plot,
    feature_names=X_train.columns,
    ax=ax, n_jobs=-1
)
plt.suptitle('Partial Dependence Plots')
\`\`\`

## Local Interpretability Methods

### SHAP (SHapley Additive exPlanations)
\`\`\`python
import shap

# Initialize SHAP explainer
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values[1], X_test, feature_names=X_train.columns)

# Individual prediction explanation
sample_idx = 0
shap.waterfall_plot(
    explainer.expected_value[1], 
    shap_values[1][sample_idx], 
    X_test.iloc[sample_idx],
    feature_names=X_train.columns
)

# Force plot for single prediction
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][sample_idx],
    X_test.iloc[sample_idx],
    feature_names=X_train.columns
)
\`\`\`

### LIME (Local Interpretable Model-agnostic Explanations)
\`\`\`python
import lime
import lime.lime_tabular

# Create LIME explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns,
    class_names=['Class 0', 'Class 1'],
    mode='classification'
)

# Explain individual prediction
instance_idx = 0
explanation = lime_explainer.explain_instance(
    X_test.iloc[instance_idx].values,
    rf_model.predict_proba,
    num_features=len(X_train.columns)
)

# Visualize explanation
explanation.show_in_notebook(show_table=True)
\`\`\`

## Deep Learning Interpretability

### Gradient-based Methods
\`\`\`python
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, Saliency, GradientShap

# PyTorch model interpretation
model = YourNeuralNetwork()
model.eval()

# Integrated Gradients
ig = IntegratedGradients(model)
attributions = ig.attribute(input_tensor, target=target_class)

# Saliency maps
saliency = Saliency(model)
grads = saliency.attribute(input_tensor, target=target_class)

# Gradient SHAP
gradient_shap = GradientShap(model)
attributions = gradient_shap.attribute(
    input_tensor, 
    baselines=baseline_tensor,
    target=target_class
)
\`\`\`

### Attention Visualization
\`\`\`python
# For transformer models
def visualize_attention(model, tokenizer, text, layer_idx=11, head_idx=0):
    inputs = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    attention = outputs.attentions[layer_idx][0, head_idx].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='Blues')
    plt.xticks(range(len(tokens)), tokens, rotation=45)
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
\`\`\`

## Fairness and Bias Detection

### Demographic Parity Analysis
\`\`\`python
def analyze_fairness(y_true, y_pred, sensitive_attribute):
    fairness_metrics = {}
    
    for group in sensitive_attribute.unique():
        group_mask = sensitive_attribute == group
        group_predictions = y_pred[group_mask]
        group_labels = y_true[group_mask]
        
        fairness_metrics[group] = {
            'selection_rate': group_predictions.mean(),
            'accuracy': (group_predictions == group_labels).mean(),
            'precision': precision_score(group_labels, group_predictions),
            'recall': recall_score(group_labels, group_predictions)
        }
    
    return fairness_metrics

# Bias mitigation strategies
def debias_model(model, X, sensitive_features):
    # Implement fairness constraints during training
    # or post-processing adjustments
    pass
\`\`\`

## Production Interpretability

### Model Cards and Documentation
\`\`\`python
class ModelCard:
    def __init__(self, model_name, version):
        self.model_name = model_name
        self.version = version
        self.metadata = {}
    
    def add_performance_metrics(self, metrics):
        self.metadata['performance'] = metrics
    
    def add_fairness_analysis(self, fairness_results):
        self.metadata['fairness'] = fairness_results
    
    def add_feature_importance(self, importance_scores):
        self.metadata['interpretability'] = importance_scores
    
    def generate_report(self):
        # Generate comprehensive model documentation
        return self.compile_model_card()
\`\`\`

## Best Practices
- **Choose appropriate methods**: Match interpretation technique to use case
- **Validate explanations**: Ensure explanations are accurate and meaningful
- **Consider stakeholders**: Tailor explanations to different audiences
- **Continuous monitoring**: Track explanation quality over time
- **Documentation**: Maintain clear records of interpretability analyses

Interpretable AI isn't just about technical methods—it's about building systems that humans can understand, trust, and effectively use.`,
    date: "2024-02-18",
    tags: ["Interpretability", "XAI", "SHAP", "Model Transparency"],
    readTime: "18 min read",
    category: "Machine Learning"
  },
  {
    id: "advanced-neural-architectures",
    title: "Advanced Neural Network Architectures: Transformers, CNNs, and Beyond",
    description: "Deep dive into modern neural network architectures including Transformers, advanced CNNs, and emerging architectures for various AI applications.",
    content: `# Advanced Neural Network Architectures: Transformers, CNNs, and Beyond

Modern neural architectures have revolutionized AI capabilities across vision, language, and multimodal tasks through innovative designs and attention mechanisms.

## Transformer Architecture Deep Dive

### Core Components
\`\`\`python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.output(context), attention_weights
\`\`\`

### Transformer Block Implementation
\`\`\`python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights
\`\`\`

### Vision Transformer (ViT)
\`\`\`python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 d_model=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, d_model)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model*4)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, d_model, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x += self.pos_embedding
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x, _ = layer(x)
        
        # Classification using class token
        x = self.norm(x)
        return self.head(x[:, 0])  # Use class token for classification
\`\`\`

## Advanced CNN Architectures

### ResNet with Squeeze-and-Excitation
\`\`\`python
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch_size, channels)
        y = self.excitation(y).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se_block = SEBlock(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se_block(out)
        out += self.shortcut(x)
        return F.relu(out)
\`\`\`

### EfficientNet-Style Architecture
\`\`\`python
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size):
        super().__init__()
        expanded_channels = in_channels * expansion_factor
        
        # Expansion phase
        self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False) if expansion_factor != 1 else nn.Identity()
        self.expand_bn = nn.BatchNorm2d(expanded_channels) if expansion_factor != 1 else nn.Identity()
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size, stride, 
            kernel_size//2, groups=expanded_channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(expanded_channels)
        
        # Squeeze-and-excitation
        self.se_block = SEBlock(expanded_channels)
        
        # Projection phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        self.use_residual = stride == 1 and in_channels == out_channels
    
    def forward(self, x):
        identity = x
        
        # Expansion
        if hasattr(self.expand_conv, 'weight'):
            x = F.relu6(self.expand_bn(self.expand_conv(x)))
        
        # Depthwise
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))
        
        # Squeeze-and-excitation
        x = self.se_block(x)
        
        # Projection
        x = self.project_bn(self.project_conv(x))
        
        # Residual connection
        if self.use_residual:
            x += identity
        
        return x
\`\`\`

## Emerging Architectures

### ConvNeXt (Modern CNN Design)
\`\`\`python
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path_rate=0.0):
        super().__init__()
        # Depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise convs
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        self.drop_path = nn.Identity()  # Implement DropPath if needed
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        x = input + self.drop_path(x)
        return x
\`\`\`

### Swin Transformer (Hierarchical Vision Transformer)
\`\`\`python
class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Create relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
\`\`\`

## Architecture Selection Guidelines

### Task-Specific Recommendations
- **Computer Vision**: CNNs for efficiency, ViTs for large datasets
- **Natural Language**: Transformers for most tasks, specialized architectures for specific domains
- **Multimodal**: Cross-attention mechanisms between modalities
- **Efficiency Requirements**: MobileNets, EfficientNets for resource constraints

### Training Strategies
- **Progressive training**: Start with smaller models, scale up
- **Transfer learning**: Leverage pre-trained models
- **Architecture search**: Automated architecture optimization
- **Ensemble methods**: Combine multiple architectures

Modern neural architectures continue evolving, with new designs emerging regularly to address specific challenges in efficiency, interpretability, and performance.`,
    date: "2024-02-12",
    tags: ["Neural Networks", "Transformers", "CNN", "Deep Learning"],
    readTime: "22 min read",
    category: "Deep Learning"
  },
  {
    id: "computer-vision-applications",
    title: "Computer Vision Applications: From Object Detection to Medical Imaging",
    description: "Comprehensive guide to modern computer vision applications including object detection, segmentation, and specialized domains like medical imaging and autonomous systems.",
    content: `# Computer Vision Applications: From Object Detection to Medical Imaging

Computer vision has transformed numerous industries through advanced deep learning techniques, enabling machines to understand and interpret visual information with human-level accuracy.

## Object Detection and Recognition

### YOLO (You Only Look Once) Implementation
\`\`\`python
import torch
import torch.nn as nn
import cv2
import numpy as np

class YOLOv5(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = self.create_backbone()
        self.neck = self.create_neck()
        self.head = self.create_head()
    
    def create_backbone(self):
        # CSPDarknet backbone
        layers = []
        # Implementation of CSP (Cross Stage Partial) connections
        return nn.Sequential(*layers)
    
    def create_neck(self):
        # PANet (Path Aggregation Network)
        return nn.ModuleList([
            # FPN layers for multi-scale feature fusion
        ])
    
    def create_head(self):
        # Detection head with anchor-based predictions
        return nn.ModuleList([
            nn.Conv2d(in_channels, (self.num_classes + 5) * 3, 1)
            for in_channels in [128, 256, 512]
        ])
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)
        
        # Multi-scale feature fusion
        fpn_features = self.neck(features)
        
        # Generate predictions at different scales
        predictions = []
        for i, head in enumerate(self.head):
            pred = head(fpn_features[i])
            predictions.append(pred)
        
        return predictions

# Non-Maximum Suppression for post-processing
def non_max_suppression(predictions, conf_threshold=0.5, iou_threshold=0.4):
    """Remove overlapping bounding boxes"""
    output = []
    
    for batch_idx, pred in enumerate(predictions):
        # Filter by confidence
        mask = pred[:, 4] > conf_threshold
        pred = pred[mask]
        
        if len(pred) == 0:
            output.append(torch.zeros((0, 6)))
            continue
        
        # Sort by confidence
        pred = pred[pred[:, 4].argsort(descending=True)]
        
        # Apply NMS
        keep = []
        while len(pred) > 0:
            keep.append(pred[0])
            
            if len(pred) == 1:
                break
            
            # Calculate IoU
            ious = calculate_iou(pred[0:1, :4], pred[1:, :4])
            pred = pred[1:][ious < iou_threshold]
        
        if len(keep) > 0:
            output.append(torch.stack(keep))
        else:
            output.append(torch.zeros((0, 6)))
    
    return output
\`\`\`

### Advanced Object Detection with Feature Pyramid Networks
\`\`\`python
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()
        
        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)
    
    def forward(self, x):
        # x is a list of feature maps at different scales
        last_inner = self.inner_blocks[-1](x[-1])
        results = [self.layer_blocks[-1](last_inner)]
        
        for i in range(len(x) - 2, -1, -1):
            inner_lateral = self.inner_blocks[i](x[i])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[i](last_inner))
        
        return results
\`\`\`

## Image Segmentation

### U-Net for Semantic Segmentation
\`\`\`python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Encoder (Contracting path)
        self.encoder1 = self.conv_block(n_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder (Expanding path)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder1 = self.conv_block(128, 64)
        
        # Final layer
        self.final_conv = nn.Conv2d(64, n_classes, 1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.final_conv(dec1)
\`\`\`

## Medical Imaging Applications

### Medical Image Preprocessing Pipeline
\`\`\`python
import pydicom
import scipy.ndimage
from skimage import measure, morphology

class MedicalImageProcessor:
    def __init__(self):
        self.hu_window_ranges = {
            'lung': (-1000, 400),
            'bone': (-200, 1000),
            'brain': (0, 80),
            'abdomen': (-150, 250)
        }
    
    def load_dicom_series(self, dicom_dir):
        """Load and sort DICOM series"""
        slices = []
        for filename in os.listdir(dicom_dir):
            if filename.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(dicom_dir, filename))
                slices.append(ds)
        
        # Sort by slice location
        slices.sort(key=lambda x: float(x.SliceLocation))
        
        # Convert to numpy array
        volume = np.stack([s.pixel_array for s in slices])
        
        # Apply rescale intercept and slope
        if hasattr(slices[0], 'RescaleIntercept'):
            intercept = slices[0].RescaleIntercept
            slope = slices[0].RescaleSlope
            volume = volume * slope + intercept
        
        return volume, slices[0]
    
    def apply_windowing(self, volume, window_type='lung'):
        """Apply HU windowing for visualization"""
        window_min, window_max = self.hu_window_ranges[window_type]
        
        volume_windowed = np.clip(volume, window_min, window_max)
        volume_windowed = (volume_windowed - window_min) / (window_max - window_min)
        
        return volume_windowed
    
    def segment_lungs(self, volume):
        """Basic lung segmentation for CT scans"""
        # Threshold for air
        binary = volume < -320
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=1000)
        
        # Fill holes
        binary = scipy.ndimage.binary_fill_holes(binary)
        
        # Label connected components
        labeled = measure.label(binary)
        
        # Keep largest components (lungs)
        regions = measure.regionprops(labeled)
        regions.sort(key=lambda x: x.area, reverse=True)
        
        lung_mask = np.zeros_like(labeled)
        for region in regions[:2]:  # Keep two largest regions
            lung_mask[labeled == region.label] = 1
        
        return lung_mask

# 3D CNN for volumetric medical image analysis
class Medical3DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        
        self.conv3d1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv3d2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3d3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout3d(0.3)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = F.relu(self.conv3d1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3d2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3d3(x))
        x = self.dropout(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
\`\`\`

## Autonomous Vehicle Vision

### Lane Detection System
\`\`\`python
class LaneDetectionPipeline:
    def __init__(self):
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.perspective_transform = None
    
    def calibrate_camera(self, calibration_images):
        """Camera calibration using chessboard pattern"""
        # Prepare object points
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane
        
        for img_path in calibration_images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        # Camera calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        self.camera_matrix = mtx
        self.distortion_coeffs = dist
    
    def undistort_image(self, img):
        """Remove camera distortion"""
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs, None, self.camera_matrix)
    
    def perspective_transform_setup(self, img_shape):
        """Setup bird's eye view transformation"""
        h, w = img_shape[:2]
        
        # Source points (trapezoid in original image)
        src = np.float32([
            [w//2 - 76, h*0.625],    # top-left
            [w//2 + 76, h*0.625],    # top-right
            [w//6, h],               # bottom-left
            [w*5//6, h]              # bottom-right
        ])
        
        # Destination points (rectangle in bird's eye view)
        dst = np.float32([
            [w//4, 0],
            [w*3//4, 0],
            [w//4, h],
            [w*3//4, h]
        ])
        
        self.perspective_transform = cv2.getPerspectiveTransform(src, dst)
        self.inverse_perspective_transform = cv2.getPerspectiveTransform(dst, src)
    
    def detect_lanes(self, img):
        """Complete lane detection pipeline"""
        # Undistort image
        undistorted = self.undistort_image(img)
        
        # Apply color and gradient thresholds
        binary = self.apply_thresholds(undistorted)
        
        # Perspective transform to bird's eye view
        warped = cv2.warpPerspective(binary, self.perspective_transform, 
                                   (img.shape[1], img.shape[0]))
        
        # Find lane lines using sliding window
        left_line, right_line = self.sliding_window_search(warped)
        
        # Fit polynomial to lane lines
        left_fit = np.polyfit(left_line[:, 1], left_line[:, 0], 2)
        right_fit = np.polyfit(right_line[:, 1], right_line[:, 0], 2)
        
        # Draw lanes back onto original image
        result = self.draw_lanes(img, left_fit, right_fit)
        
        return result, left_fit, right_fit
    
    def apply_thresholds(self, img):
        """Apply color and gradient thresholds"""
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]
        
        # Sobel gradient threshold
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        
        # Binary thresholds
        sobel_binary = np.zeros_like(scaled_sobel)
        sobel_binary[(scaled_sobel >= 20) & (scaled_sobel <= 100)] = 1
        
        # Color threshold
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= 170) & (s_channel <= 255)] = 1
        
        # Combine thresholds
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1
        
        return combined_binary
\`\`\`

## Best Practices and Optimization

### Data Augmentation for Computer Vision
\`\`\`python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_training_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.2),
        A.OneOf([
            A.IAAAdditiveGaussianNoise(),
            A.GaussNoise(),
        ], p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.OneOf([
            A.CLAHE(clip_limit=2),
            A.IAASharpen(),
            A.IAAEmboss(),
        ], p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
\`\`\`

### Model Optimization Techniques
- **Quantization**: Reduce model precision for faster inference
- **Pruning**: Remove unnecessary network connections
- **Knowledge Distillation**: Train smaller models to mimic larger ones
- **TensorRT/ONNX**: Optimize for specific hardware

Computer vision applications continue expanding across industries, from healthcare and automotive to manufacturing and entertainment, driven by advancing deep learning architectures and increasing computational power.`,
    date: "2024-02-08",
    tags: ["Computer Vision", "Object Detection", "Medical Imaging", "Autonomous Systems"],
    readTime: "25 min read",
    category: "Deep Learning"
  },
  {
    id: "llmops-production-systems",
    title: "LLMOps: Production Systems for Large Language Models",
    description: "Comprehensive guide to deploying and managing large language models in production, covering infrastructure, monitoring, and optimization strategies.",
    content: `# LLMOps: Production Systems for Large Language Models

LLMOps encompasses the practices and infrastructure needed to deploy, monitor, and maintain large language models in production environments at scale.

## LLM Infrastructure Architecture

### Model Serving Infrastructure
\`\`\`python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
import uvloop
from concurrent.futures import ThreadPoolExecutor

class LLMServingEngine:
    def __init__(self, model_name, device="cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True  # Quantization for memory efficiency
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def generate_async(self, prompt, max_length=512, temperature=0.7):
        """Asynchronous text generation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            self._generate_sync, 
            prompt, max_length, temperature
        )
    
    def _generate_sync(self, prompt, max_length, temperature):
        """Synchronous generation for thread pool"""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):]  # Remove input prompt from output

# FastAPI application
app = FastAPI(title="LLM API", version="1.0.0")
llm_engine = LLMServingEngine("microsoft/DialoGPT-medium")

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 512
    temperature: float = 0.7
    stream: bool = False

class GenerationResponse(BaseModel):
    generated_text: str
    metadata: dict

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        generated_text = await llm_engine.generate_async(
            request.prompt, 
            request.max_length, 
            request.temperature
        )
        
        return GenerationResponse(
            generated_text=generated_text,
            metadata={
                "model": "microsoft/DialoGPT-medium",
                "prompt_length": len(request.prompt),
                "generated_length": len(generated_text)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
\`\`\`

### Distributed Model Serving with Ray
\`\`\`python
import ray
from ray import serve
import torch

@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_concurrent_queries=10,
    autoscaling_config={"min_replicas": 1, "max_replicas": 4}
)
class DistributedLLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    async def __call__(self, request):
        prompt = request.json()["prompt"]
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").cuda()
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"generated_text": response}

# Deploy the service
ray.init()
serve.start()

llm_service = DistributedLLMService.bind("gpt2")
serve.run(llm_service, name="llm-service", route_prefix="/generate")
\`\`\`

## Model Optimization and Quantization

### Dynamic Quantization
\`\`\`python
import torch.quantization

class ModelOptimizer:
    def __init__(self, model):
        self.original_model = model
        
    def apply_dynamic_quantization(self):
        """Apply dynamic quantization for inference speedup"""
        quantized_model = torch.quantization.quantize_dynamic(
            self.original_model,
            {torch.nn.Linear},  # Specify layers to quantize
            dtype=torch.qint8
        )
        return quantized_model
    
    def apply_static_quantization(self, calibration_dataloader):
        """Apply static quantization with calibration data"""
        # Prepare model for quantization
        self.original_model.eval()
        self.original_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self.original_model, inplace=True)
        
        # Calibrate with representative data
        with torch.no_grad():
            for batch in calibration_dataloader:
                self.original_model(batch)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.original_model, inplace=False)
        return quantized_model

# Model pruning for efficiency
import torch.nn.utils.prune as prune

def apply_structured_pruning(model, pruning_ratio=0.2):
    """Apply structured pruning to reduce model size"""
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
            prune.remove(module, 'weight')
    
    return model
\`\`\`

### Model Compilation and Optimization
\`\`\`python
import torch._dynamo as dynamo
import torch.compile

class OptimizedModelWrapper:
    def __init__(self, model, optimization_level="default"):
        self.model = model
        self.optimization_level = optimization_level
        self._compile_model()
    
    def _compile_model(self):
        """Compile model for optimized inference"""
        if self.optimization_level == "aggressive":
            # Aggressive optimization for maximum speed
            self.compiled_model = torch.compile(
                self.model,
                mode="max-autotune",
                dynamic=True
            )
        elif self.optimization_level == "balanced":
            # Balanced optimization
            self.compiled_model = torch.compile(
                self.model,
                mode="default",
                dynamic=False
            )
        else:
            # Conservative optimization
            self.compiled_model = torch.compile(
                self.model,
                mode="reduce-overhead"
            )
    
    def generate(self, *args, **kwargs):
        return self.compiled_model.generate(*args, **kwargs)
\`\`\`

## Monitoring and Observability

### Performance Monitoring System
\`\`\`python
import time
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging

class LLMMetricsCollector:
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'llm_requests_total', 
            'Total LLM requests',
            ['model', 'endpoint']
        )
        
        self.request_duration = Histogram(
            'llm_request_duration_seconds',
            'LLM request duration',
            ['model', 'endpoint']
        )
        
        self.gpu_utilization = Gauge(
            'llm_gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.memory_usage = Gauge(
            'llm_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.token_throughput = Gauge(
            'llm_tokens_per_second',
            'Token generation throughput'
        )
        
        # Start Prometheus metrics server
        start_http_server(8000)
    
    def record_request(self, model_name, endpoint, duration, token_count):
        """Record request metrics"""
        self.request_count.labels(model=model_name, endpoint=endpoint).inc()
        self.request_duration.labels(model=model_name, endpoint=endpoint).observe(duration)
        
        if duration > 0:
            self.token_throughput.set(token_count / duration)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        # GPU metrics
        gpus = GPUtil.getGPUs()
        if gpus:
            self.gpu_utilization.set(gpus[0].load * 100)
        
        # Memory metrics
        memory_info = psutil.virtual_memory()
        self.memory_usage.set(memory_info.used)

# Middleware for automatic metrics collection
class MetricsMiddleware:
    def __init__(self, app, metrics_collector):
        self.app = app
        self.metrics = metrics_collector
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            
            # Process request
            await self.app(scope, receive, send)
            
            # Record metrics
            duration = time.time() - start_time
            endpoint = scope.get("path", "unknown")
            
            # Extract token count from response (implementation specific)
            token_count = self._extract_token_count(scope)
            
            self.metrics.record_request("llm-model", endpoint, duration, token_count)
            self.metrics.update_system_metrics()
\`\`\`

### Quality Monitoring and A/B Testing
\`\`\`python
import hashlib
import random
from typing import Dict, Any

class ModelQualityMonitor:
    def __init__(self):
        self.quality_metrics = {}
        self.response_cache = {}
        
    def evaluate_response_quality(self, prompt: str, response: str) -> Dict[str, float]:
        """Evaluate response quality using multiple metrics"""
        metrics = {}
        
        # Length-based metrics
        metrics['response_length'] = len(response.split())
        metrics['prompt_response_ratio'] = len(response.split()) / len(prompt.split())
        
        # Coherence check (simplified)
        metrics['coherence_score'] = self._calculate_coherence(response)
        
        # Toxicity detection (placeholder - use actual toxicity model)
        metrics['toxicity_score'] = self._detect_toxicity(response)
        
        # Factual consistency (placeholder)
        metrics['factual_score'] = self._check_factual_consistency(prompt, response)
        
        return metrics
    
    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score"""
        sentences = text.split('.')
        if len(sentences) < 2:
            return 1.0
        
        # Simplified coherence calculation
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        coherence = min(1.0, avg_sentence_length / 20.0)  # Normalize
        
        return coherence
    
    def _detect_toxicity(self, text: str) -> float:
        """Detect toxicity in generated text"""
        # Placeholder - integrate with actual toxicity detection model
        toxic_words = ['hate', 'violence', 'toxic', 'harmful']
        toxic_count = sum(1 for word in toxic_words if word in text.lower())
        return min(1.0, toxic_count / 10.0)
    
    def _check_factual_consistency(self, prompt: str, response: str) -> float:
        """Check factual consistency of response"""
        # Placeholder - integrate with fact-checking model
        return random.uniform(0.7, 1.0)  # Mock score

class ABTestingFramework:
    def __init__(self, models: Dict[str, Any]):
        self.models = models
        self.traffic_split = {model: 1.0/len(models) for model in models}
        self.performance_metrics = {model: [] for model in models}
    
    def select_model(self, user_id: str) -> str:
        """Select model based on user hash and traffic split"""
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        threshold = user_hash / (2**128)
        
        cumulative = 0
        for model, split in self.traffic_split.items():
            cumulative += split
            if threshold <= cumulative:
                return model
        
        return list(self.models.keys())[0]  # Fallback
    
    def update_traffic_split(self, performance_data: Dict[str, float]):
        """Update traffic split based on performance"""
        best_model = max(performance_data.keys(), key=lambda k: performance_data[k])
        
        # Gradually shift traffic to better performing model
        for model in self.traffic_split:
            if model == best_model:
                self.traffic_split[model] = min(0.8, self.traffic_split[model] + 0.1)
            else:
                self.traffic_split[model] = max(0.1, self.traffic_split[model] - 0.05)
        
        # Normalize
        total = sum(self.traffic_split.values())
        self.traffic_split = {k: v/total for k, v in self.traffic_split.items()}
\`\`\`

## Security and Compliance

### Content Filtering and Safety
\`\`\`python
import re
from typing import List, Tuple

class ContentSafetyFilter:
    def __init__(self):
        self.blocked_patterns = [
            r'\\b(?:password|api[_\\s]?key|secret|token)\\b',
            r'\\b(?:hack|exploit|vulnerability)\\b',
            r'\\b(?:illegal|fraud|scam)\\b'
        ]
        
        self.pii_patterns = [
            r'\\b\\d{3}-\\d{2}-\\d{4}\\b',  # SSN
            r'\\b\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}[\\s-]?\\d{4}\\b',  # Credit card
            r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'  # Email
        ]
    
    def check_input_safety(self, text: str) -> Tuple[bool, List[str]]:
        """Check if input text is safe"""
        violations = []
        
        # Check for blocked content
        for pattern in self.blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(f"Blocked content detected: {pattern}")
        
        # Check for PII
        for pattern in self.pii_patterns:
            if re.search(pattern, text):
                violations.append(f"PII detected: {pattern}")
        
        is_safe = len(violations) == 0
        return is_safe, violations
    
    def sanitize_output(self, text: str) -> str:
        """Remove sensitive information from output"""
        sanitized = text
        
        # Remove potential PII
        for pattern in self.pii_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized)
        
        return sanitized

# Audit logging
class AuditLogger:
    def __init__(self, log_file="llm_audit.log"):
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def log_request(self, user_id: str, prompt: str, response: str, metadata: dict):
        """Log LLM request for audit purposes"""
        audit_entry = {
            "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:16],  # Anonymized
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "response_length": len(response),
            "model": metadata.get("model", "unknown"),
            "timestamp": time.time()
        }
        
        self.logger.info(f"LLM_REQUEST: {audit_entry}")
\`\`\`

## Cost Optimization Strategies

### Request Batching and Caching
\`\`\`python
import asyncio
from collections import defaultdict
import hashlib
import time

class RequestBatcher:
    def __init__(self, batch_size=8, max_wait_time=0.1):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.request_futures = {}
        
    async def add_request(self, prompt: str, **kwargs):
        """Add request to batch"""
        request_id = hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()
        future = asyncio.Future()
        
        self.pending_requests.append({
            'id': request_id,
            'prompt': prompt,
            'kwargs': kwargs,
            'future': future,
            'timestamp': time.time()
        })
        
        # Trigger batch processing if needed
        if len(self.pending_requests) >= self.batch_size:
            asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests[:self.batch_size]
        self.pending_requests = self.pending_requests[self.batch_size:]
        
        # Batch generation (implementation specific)
        prompts = [req['prompt'] for req in batch]
        results = await self._generate_batch(prompts)
        
        # Resolve futures
        for request, result in zip(batch, results):
            request['future'].set_result(result)
    
    async def _generate_batch(self, prompts: List[str]) -> List[str]:
        """Generate responses for batch of prompts"""
        # Implementation depends on model and framework
        # This is a placeholder
        return [f"Response to: {prompt}" for prompt in prompts]

class ResponseCache:
    def __init__(self, max_size=10000, ttl=3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, prompt: str, **kwargs) -> str:
        """Get cached response"""
        cache_key = self._generate_key(prompt, **kwargs)
        
        if cache_key in self.cache:
            # Check TTL
            if time.time() - self.access_times[cache_key] < self.ttl:
                self.access_times[cache_key] = time.time()
                return self.cache[cache_key]
            else:
                # Expired
                del self.cache[cache_key]
                del self.access_times[cache_key]
        
        return None
    
    def set(self, prompt: str, response: str, **kwargs):
        """Cache response"""
        cache_key = self._generate_key(prompt, **kwargs)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[cache_key] = response
        self.access_times[cache_key] = time.time()
    
    def _generate_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key from prompt and parameters"""
        key_data = f"{prompt}{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
\`\`\`

LLMOps requires careful attention to infrastructure, monitoring, safety, and cost optimization to successfully deploy and maintain large language models at production scale.`,
    date: "2024-02-05",
    tags: ["LLMOps", "Model Deployment", "Infrastructure", "Monitoring"],
    readTime: "28 min read",
    category: "AI Agents"
  },
  {
    id: "ai-agent-orchestration",
    title: "AI Agent Orchestration: Building Intelligent Workflow Systems",
    description: "Advanced techniques for orchestrating multiple AI agents to work together, including workflow management, communication protocols, and coordination strategies.",
    content: `# AI Agent Orchestration: Building Intelligent Workflow Systems

Agent orchestration enables complex AI systems where multiple specialized agents collaborate to solve problems that would be difficult for a single agent to handle effectively.

## Orchestration Architecture Patterns

### Central Orchestrator Pattern
\`\`\`python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import asyncio
import logging

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    id: str
    type: str
    input_data: Dict[str, Any]
    agent_id: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = None
    retry_count: int = 0
    max_retries: int = 3

class CentralOrchestrator:
    def __init__(self):
        self.agents = {}
        self.task_queue = asyncio.Queue()
        self.completed_tasks = {}
        self.workflow_graph = {}
        self.logger = logging.getLogger(__name__)
    
    def register_agent(self, agent_id: str, agent: 'BaseAgent'):
        """Register an agent with the orchestrator"""
        self.agents[agent_id] = agent
        self.logger.info(f"Registered agent: {agent_id}")
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete workflow"""
        workflow_id = workflow_definition['id']
        tasks = workflow_definition['tasks']
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tasks)
        
        # Execute tasks in topological order
        results = {}
        ready_tasks = self._get_ready_tasks(dependency_graph, set())
        
        while ready_tasks or self._has_pending_tasks():
            # Execute ready tasks
            task_futures = []
            for task_def in ready_tasks:
                task = Task(
                    id=task_def['id'],
                    type=task_def['type'],
                    input_data=task_def.get('input', {}),
                    dependencies=task_def.get('dependencies', [])
                )
                
                # Merge dependency results into input
                for dep_id in task.dependencies:
                    if dep_id in results:
                        task.input_data.update(results[dep_id])
                
                future = asyncio.create_task(self._execute_task(task))
                task_futures.append((task.id, future))
            
            # Wait for task completion
            for task_id, future in task_futures:
                try:
                    result = await future
                    results[task_id] = result
                    self.logger.info(f"Task {task_id} completed successfully")
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {str(e)}")
                    results[task_id] = {"error": str(e)}
            
            # Get next ready tasks
            completed_task_ids = set(results.keys())
            ready_tasks = self._get_ready_tasks(dependency_graph, completed_task_ids)
        
        return results
    
    async def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a single task"""
        # Select appropriate agent
        agent = self._select_agent_for_task(task)
        
        if not agent:
            raise Exception(f"No suitable agent found for task type: {task.type}")
        
        task.status = TaskStatus.IN_PROGRESS
        task.agent_id = agent.agent_id
        
        try:
            # Execute task with retry logic
            for attempt in range(task.max_retries + 1):
                try:
                    result = await agent.execute(task.input_data)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    return result
                except Exception as e:
                    if attempt < task.max_retries:
                        task.retry_count += 1
                        self.logger.warning(f"Task {task.id} attempt {attempt + 1} failed, retrying...")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
        
        except Exception as e:
            task.status = TaskStatus.FAILED
            raise e
    
    def _select_agent_for_task(self, task: Task) -> Optional['BaseAgent']:
        """Select the best agent for a given task"""
        suitable_agents = [
            agent for agent in self.agents.values()
            if task.type in agent.capabilities
        ]
        
        if not suitable_agents:
            return None
        
        # Simple selection - can be enhanced with load balancing
        return min(suitable_agents, key=lambda a: len(a.current_tasks))
    
    def _build_dependency_graph(self, tasks: List[Dict]) -> Dict[str, List[str]]:
        """Build task dependency graph"""
        graph = {}
        for task in tasks:
            graph[task['id']] = task.get('dependencies', [])
        return graph
    
    def _get_ready_tasks(self, graph: Dict[str, List[str]], completed: set) -> List[Dict]:
        """Get tasks that are ready to execute"""
        ready = []
        for task_id, dependencies in graph.items():
            if task_id not in completed and all(dep in completed for dep in dependencies):
                ready.append({'id': task_id, 'dependencies': dependencies})
        return ready
\`\`\`

### Event-Driven Orchestration
\`\`\`python
import asyncio
from typing import Callable, Dict, Any
from dataclasses import dataclass, field
import json

@dataclass
class Event:
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    correlation_id: str = ""

class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.event_history = []
        self.middleware = []
    
    def subscribe(self, event_type: str, handler: Callable[[Event], Any]):
        """Subscribe to events of a specific type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(handler)
    
    async def publish(self, event: Event):
        """Publish an event to all subscribers"""
        # Apply middleware
        for middleware in self.middleware:
            event = await middleware(event)
        
        # Store event history
        self.event_history.append(event)
        
        # Notify subscribers
        handlers = self.subscribers.get(event.type, [])
        if handlers:
            tasks = [asyncio.create_task(handler(event)) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def add_middleware(self, middleware: Callable[[Event], Event]):
        """Add middleware for event processing"""
        self.middleware.append(middleware)

class EventDrivenOrchestrator:
    def __init__(self):
        self.event_bus = EventBus()
        self.agents = {}
        self.workflow_state = {}
        
        # Setup event handlers
        self.event_bus.subscribe("task_completed", self._handle_task_completion)
        self.event_bus.subscribe("task_failed", self._handle_task_failure)
        self.event_bus.subscribe("workflow_started", self._handle_workflow_start)
    
    async def start_workflow(self, workflow_id: str, initial_data: Dict[str, Any]):
        """Start a new workflow"""
        self.workflow_state[workflow_id] = {
            'status': 'running',
            'completed_tasks': set(),
            'failed_tasks': set(),
            'data': initial_data
        }
        
        await self.event_bus.publish(Event(
            type="workflow_started",
            data={'workflow_id': workflow_id, 'initial_data': initial_data}
        ))
    
    async def _handle_task_completion(self, event: Event):
        """Handle task completion events"""
        workflow_id = event.data['workflow_id']
        task_id = event.data['task_id']
        result = event.data['result']
        
        # Update workflow state
        self.workflow_state[workflow_id]['completed_tasks'].add(task_id)
        self.workflow_state[workflow_id]['data'].update(result)
        
        # Check for next tasks to execute
        next_tasks = self._get_next_tasks(workflow_id, task_id)
        for next_task in next_tasks:
            await self._trigger_task(workflow_id, next_task)
    
    async def _handle_task_failure(self, event: Event):
        """Handle task failure events"""
        workflow_id = event.data['workflow_id']
        task_id = event.data['task_id']
        error = event.data['error']
        
        self.workflow_state[workflow_id]['failed_tasks'].add(task_id)
        
        # Implement failure recovery strategies
        await self._handle_failure_recovery(workflow_id, task_id, error)
    
    async def _trigger_task(self, workflow_id: str, task_definition: Dict[str, Any]):
        """Trigger execution of a task"""
        task_id = task_definition['id']
        agent_type = task_definition['agent_type']
        
        # Find suitable agent
        agent = self._find_agent(agent_type)
        if not agent:
            await self.event_bus.publish(Event(
                type="task_failed",
                data={
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'error': f'No agent available for type: {agent_type}'
                }
            ))
            return
        
        # Execute task
        try:
            input_data = self.workflow_state[workflow_id]['data']
            result = await agent.execute(input_data)
            
            await self.event_bus.publish(Event(
                type="task_completed",
                data={
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'result': result
                }
            ))
        
        except Exception as e:
            await self.event_bus.publish(Event(
                type="task_failed",
                data={
                    'workflow_id': workflow_id,
                    'task_id': task_id,
                    'error': str(e)
                }
            ))
\`\`\`

## Agent Communication Protocols

### Message-Based Communication
\`\`\`python
from abc import ABC, abstractmethod
import asyncio
import json
from typing import Optional

class Message:
    def __init__(self, sender: str, receiver: str, content: Any, message_type: str = "data"):
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.message_type = message_type
        self.timestamp = time.time()
        self.id = str(uuid.uuid4())

class CommunicationProtocol(ABC):
    @abstractmethod
    async def send_message(self, message: Message):
        pass
    
    @abstractmethod
    async def receive_message(self) -> Optional[Message]:
        pass

class MessageQueue(CommunicationProtocol):
    def __init__(self):
        self.queues = {}
        self.message_handlers = {}
    
    def get_queue(self, agent_id: str) -> asyncio.Queue:
        if agent_id not in self.queues:
            self.queues[agent_id] = asyncio.Queue()
        return self.queues[agent_id]
    
    async def send_message(self, message: Message):
        """Send message to recipient's queue"""
        receiver_queue = self.get_queue(message.receiver)
        await receiver_queue.put(message)
    
    async def receive_message(self, agent_id: str) -> Optional[Message]:
        """Receive message from agent's queue"""
        queue = self.get_queue(agent_id)
        try:
            return await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None
    
    def register_handler(self, agent_id: str, message_type: str, handler: Callable):
        """Register message handler for specific message types"""
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = {}
        self.message_handlers[agent_id][message_type] = handler

class P2PCommunication(CommunicationProtocol):
    def __init__(self):
        self.connections = {}
        self.message_buffer = {}
    
    def establish_connection(self, agent1: str, agent2: str):
        """Establish direct connection between two agents"""
        if agent1 not in self.connections:
            self.connections[agent1] = set()
        if agent2 not in self.connections:
            self.connections[agent2] = set()
        
        self.connections[agent1].add(agent2)
        self.connections[agent2].add(agent1)
    
    async def send_message(self, message: Message):
        """Send direct message between connected agents"""
        if message.receiver in self.connections.get(message.sender, set()):
            if message.receiver not in self.message_buffer:
                self.message_buffer[message.receiver] = []
            self.message_buffer[message.receiver].append(message)
        else:
            raise Exception(f"No connection between {message.sender} and {message.receiver}")
    
    async def receive_message(self, agent_id: str) -> Optional[Message]:
        """Receive direct message"""
        if agent_id in self.message_buffer and self.message_buffer[agent_id]:
            return self.message_buffer[agent_id].pop(0)
        return None
\`\`\`

### Consensus and Coordination Mechanisms
\`\`\`python
class ConsensusManager:
    def __init__(self, agents: List[str]):
        self.agents = agents
        self.proposals = {}
        self.votes = {}
        
    async def propose_decision(self, proposal_id: str, proposal_data: Dict[str, Any]) -> bool:
        """Propose a decision that requires consensus"""
        self.proposals[proposal_id] = proposal_data
        self.votes[proposal_id] = {}
        
        # Collect votes from all agents
        vote_tasks = []
        for agent_id in self.agents:
            task = asyncio.create_task(
                self._collect_vote(agent_id, proposal_id, proposal_data)
            )
            vote_tasks.append(task)
        
        # Wait for all votes
        await asyncio.gather(*vote_tasks)
        
        # Calculate consensus
        return self._calculate_consensus(proposal_id)
    
    async def _collect_vote(self, agent_id: str, proposal_id: str, proposal_data: Dict[str, Any]):
        """Collect vote from an agent"""
        agent = self.get_agent(agent_id)
        vote = await agent.vote_on_proposal(proposal_data)
        self.votes[proposal_id][agent_id] = vote
    
    def _calculate_consensus(self, proposal_id: str) -> bool:
        """Calculate if consensus is reached"""
        votes = self.votes[proposal_id]
        total_votes = len(votes)
        positive_votes = sum(1 for vote in votes.values() if vote)
        
        # Require simple majority
        return positive_votes > total_votes / 2

class CoordinationProtocol:
    def __init__(self):
        self.locks = {}
        self.resource_assignments = {}
        
    async def acquire_resource_lock(self, resource_id: str, agent_id: str, timeout: float = 30.0) -> bool:
        """Acquire exclusive lock on a resource"""
        if resource_id in self.locks:
            # Resource is locked
            try:
                await asyncio.wait_for(self.locks[resource_id].wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return False
        
        # Create lock for this resource
        self.locks[resource_id] = asyncio.Event()
        self.resource_assignments[resource_id] = agent_id
        return True
    
    def release_resource_lock(self, resource_id: str, agent_id: str):
        """Release lock on a resource"""
        if (resource_id in self.resource_assignments and 
            self.resource_assignments[resource_id] == agent_id):
            
            if resource_id in self.locks:
                self.locks[resource_id].set()
                del self.locks[resource_id]
            
            del self.resource_assignments[resource_id]
    
    async def coordinate_task_allocation(self, tasks: List[Dict], agents: List[str]) -> Dict[str, str]:
        """Coordinate allocation of tasks to agents"""
        allocation = {}
        
        # Simple round-robin allocation
        for i, task in enumerate(tasks):
            agent_id = agents[i % len(agents)]
            allocation[task['id']] = agent_id
        
        return allocation
\`\`\`

## Workflow Management

### Dynamic Workflow Adaptation
\`\`\`python
class AdaptiveWorkflowManager:
    def __init__(self):
        self.workflow_templates = {}
        self.active_workflows = {}
        self.performance_metrics = {}
        
    def register_workflow_template(self, template_id: str, template: Dict[str, Any]):
        """Register a workflow template"""
        self.workflow_templates[template_id] = template
    
    async def execute_adaptive_workflow(self, template_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with adaptive modifications"""
        template = self.workflow_templates[template_id]
        workflow_id = f"{template_id}_{int(time.time())}"
        
        # Create workflow instance
        workflow = self._instantiate_workflow(template, input_data)
        self.active_workflows[workflow_id] = workflow
        
        # Execute with monitoring and adaptation
        results = {}
        current_step = 0
        
        while current_step < len(workflow['steps']):
            step = workflow['steps'][current_step]
            
            # Monitor performance and adapt if necessary
            adaptation = await self._check_adaptation_needed(workflow_id, step)
            if adaptation:
                workflow = await self._apply_adaptation(workflow, adaptation)
            
            # Execute step
            step_result = await self._execute_workflow_step(step, results)
            results[step['id']] = step_result
            
            # Determine next step (may skip or branch)
            current_step = self._determine_next_step(workflow, current_step, step_result)
        
        return results
    
    async def _check_adaptation_needed(self, workflow_id: str, step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if workflow adaptation is needed"""
        # Monitor performance metrics
        performance = await self._get_performance_metrics(workflow_id)
        
        if performance['latency'] > step.get('max_latency', float('inf')):
            return {
                'type': 'performance_optimization',
                'action': 'scale_resources',
                'target_step': step['id']
            }
        
        if performance['error_rate'] > step.get('max_error_rate', 1.0):
            return {
                'type': 'error_recovery',
                'action': 'add_retry_logic',
                'target_step': step['id']
            }
        
        return None
    
    async def _apply_adaptation(self, workflow: Dict[str, Any], adaptation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptation to workflow"""
        if adaptation['type'] == 'performance_optimization':
            # Scale resources for target step
            for step in workflow['steps']:
                if step['id'] == adaptation['target_step']:
                    step['resources'] = step.get('resources', {})
                    step['resources']['replicas'] = step['resources'].get('replicas', 1) * 2
        
        elif adaptation['type'] == 'error_recovery':
            # Add retry logic
            for step in workflow['steps']:
                if step['id'] == adaptation['target_step']:
                    step['retry_policy'] = {
                        'max_retries': 3,
                        'backoff_strategy': 'exponential'
                    }
        
        return workflow
    
    def _determine_next_step(self, workflow: Dict[str, Any], current_step: int, step_result: Dict[str, Any]) -> int:
        """Determine next step based on current step result"""
        step = workflow['steps'][current_step]
        
        # Check for conditional branching
        if 'conditions' in step:
            for condition in step['conditions']:
                if self._evaluate_condition(condition, step_result):
                    return condition['next_step']
        
        # Default: next sequential step
        return current_step + 1
    
    def _evaluate_condition(self, condition: Dict[str, Any], step_result: Dict[str, Any]) -> bool:
        """Evaluate a conditional expression"""
        expression = condition['expression']
        # Simple expression evaluation (can be enhanced)
        return eval(expression, {"result": step_result})
\`\`\`

## Error Handling and Recovery

### Fault-Tolerant Orchestration
\`\`\`python
class FaultTolerantOrchestrator:
    def __init__(self):
        self.failure_handlers = {}
        self.recovery_strategies = {}
        self.circuit_breakers = {}
    
    def register_failure_handler(self, error_type: str, handler: Callable):
        """Register handler for specific error types"""
        self.failure_handlers[error_type] = handler
    
    async def execute_with_fault_tolerance(self, task: Task) -> Dict[str, Any]:
        """Execute task with comprehensive fault tolerance"""
        agent_id = task.agent_id
        
        # Check circuit breaker
        if self._is_circuit_open(agent_id):
            raise Exception(f"Circuit breaker open for agent {agent_id}")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                self._execute_task_with_monitoring(task),
                timeout=task.get('timeout', 300.0)
            )
            
            # Reset circuit breaker on success
            self._reset_circuit_breaker(agent_id)
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(agent_id, str(e))
            
            # Apply recovery strategy
            recovery_result = await self._apply_recovery_strategy(task, e)
            if recovery_result:
                return recovery_result
            
            # If recovery fails, trigger circuit breaker
            self._trigger_circuit_breaker(agent_id)
            raise e
    
    async def _apply_recovery_strategy(self, task: Task, error: Exception) -> Optional[Dict[str, Any]]:
        """Apply appropriate recovery strategy"""
        error_type = type(error).__name__
        
        if error_type in self.recovery_strategies:
            strategy = self.recovery_strategies[error_type]
            
            if strategy == 'retry':
                return await self._retry_with_backoff(task)
            elif strategy == 'failover':
                return await self._failover_to_backup_agent(task)
            elif strategy == 'graceful_degradation':
                return await self._graceful_degradation(task)
        
        return None
    
    async def _retry_with_backoff(self, task: Task, max_retries: int = 3) -> Optional[Dict[str, Any]]:
        """Retry task with exponential backoff"""
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                return await self._execute_task_with_monitoring(task)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
        return None
    
    async def _failover_to_backup_agent(self, task: Task) -> Optional[Dict[str, Any]]:
        """Failover to a backup agent"""
        backup_agents = self._get_backup_agents(task.agent_id)
        
        for backup_agent_id in backup_agents:
            try:
                task.agent_id = backup_agent_id
                return await self._execute_task_with_monitoring(task)
            except Exception:
                continue
        
        return None
    
    def _is_circuit_open(self, agent_id: str) -> bool:
        """Check if circuit breaker is open for an agent"""
        if agent_id not in self.circuit_breakers:
            return False
        
        cb = self.circuit_breakers[agent_id]
        if cb['state'] == 'open':
            # Check if enough time has passed to try again
            if time.time() - cb['opened_at'] > cb['timeout']:
                cb['state'] = 'half_open'
                return False
            return True
        
        return False
\`\`\`

Effective agent orchestration requires careful design of communication protocols, coordination mechanisms, and fault tolerance strategies to build robust, scalable AI systems that can handle complex, multi-step workflows.`,
    date: "2024-02-01",
    tags: ["AI Orchestration", "Multi-Agent Systems", "Workflow Management"],
    readTime: "24 min read",
    category: "AI Agents"
  },
  {
    id: "distributed-agentic-systems",
    title: "Building Distributed Agentic AI Systems: Architecture and Implementation",
    description: "Comprehensive guide to designing and implementing distributed agentic AI systems that can scale across multiple nodes and handle complex collaborative tasks.",
    content: `# Building Distributed Agentic AI Systems

Distributed agentic AI systems enable scalable, fault-tolerant AI applications by distributing intelligence across multiple nodes, each capable of autonomous decision-making and collaboration.

## Distributed Architecture Fundamentals

### Node-Based Architecture
\`\`\`python
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

@dataclass
class NodeInfo:
    node_id: str
    node_type: str
    capabilities: List[str]
    current_load: float
    max_capacity: int
    network_address: str
    health_status: str = "healthy"

class DistributedNode(ABC):
    def __init__(self, node_id: str, node_type: str, capabilities: List[str]):
        self.node_info = NodeInfo(
            node_id=node_id,
            node_type=node_type,
            capabilities=capabilities,
            current_load=0.0,
            max_capacity=100,
            network_address=f"http://localhost:8000/{node_id}"
        )
        self.peer_nodes = {}
        self.message_handlers = {}
        self.task_queue = asyncio.Queue()
        self.is_running = False
    
    async def start(self):
        """Start the distributed node"""
        self.is_running = True
        await asyncio.gather(
            self._process_tasks(),
            self._heartbeat_monitor(),
            self._handle_network_messages()
        )
    
    async def stop(self):
        """Stop the distributed node"""
        self.is_running = False
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task on this node"""
        pass
    
    async def _process_tasks(self):
        """Process tasks from the queue"""
        while self.is_running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                result = await self.process_task(task)
                await self._send_result(task['requester'], result)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                await self._handle_task_error(task, e)
    
    async def _heartbeat_monitor(self):
        """Monitor node health and send heartbeats"""
        while self.is_running:
            await self._update_health_metrics()
            await self._broadcast_heartbeat()
            await asyncio.sleep(30)  # Heartbeat every 30 seconds
    
    async def _update_health_metrics(self):
        """Update node health and load metrics"""
        # Calculate current load
        queue_size = self.task_queue.qsize()
        self.node_info.current_load = min(100.0, (queue_size / self.node_info.max_capacity) * 100)
        
        # Update health status based on system metrics
        if self.node_info.current_load > 90:
            self.node_info.health_status = "overloaded"
        elif self.node_info.current_load > 70:
            self.node_info.health_status = "busy"
        else:
            self.node_info.health_status = "healthy"

class DistributedTaskScheduler:
    def __init__(self):
        self.nodes = {}
        self.task_history = {}
        self.load_balancer = LoadBalancer()
        
    def register_node(self, node: DistributedNode):
        """Register a node with the scheduler"""
        self.nodes[node.node_info.node_id] = node
        self.load_balancer.add_node(node.node_info)
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for distributed execution"""
        task_id = str(uuid.uuid4())
        task['id'] = task_id
        task['submitted_at'] = time.time()
        
        # Select optimal node for task
        selected_node = await self.load_balancer.select_node(task)
        
        if not selected_node:
            raise Exception("No available nodes for task execution")
        
        # Submit task to selected node
        await selected_node.task_queue.put(task)
        
        # Track task
        self.task_history[task_id] = {
            'task': task,
            'assigned_node': selected_node.node_info.node_id,
            'status': 'submitted'
        }
        
        return task_id

class LoadBalancer:
    def __init__(self):
        self.balancing_strategy = "least_loaded"
        self.node_weights = {}
    
    async def select_node(self, task: Dict[str, Any]) -> Optional[DistributedNode]:
        """Select the best node for a task"""
        required_capabilities = task.get('required_capabilities', [])
        
        # Filter nodes by capabilities
        suitable_nodes = [
            node for node in self.nodes.values()
            if all(cap in node.node_info.capabilities for cap in required_capabilities)
            and node.node_info.health_status != "offline"
        ]
        
        if not suitable_nodes:
            return None
        
        # Apply load balancing strategy
        if self.balancing_strategy == "least_loaded":
            return min(suitable_nodes, key=lambda n: n.node_info.current_load)
        elif self.balancing_strategy == "round_robin":
            return self._round_robin_selection(suitable_nodes)
        elif self.balancing_strategy == "weighted":
            return self._weighted_selection(suitable_nodes, task)
        
        return suitable_nodes[0]  # Fallback
    
    def _weighted_selection(self, nodes: List[DistributedNode], task: Dict[str, Any]) -> DistributedNode:
        """Select node based on weighted scoring"""
        scores = {}
        
        for node in nodes:
            score = 0
            
            # Load factor (lower is better)
            score += (100 - node.node_info.current_load) * 0.4
            
            # Capability match score
            required_caps = set(task.get('required_capabilities', []))
            node_caps = set(node.node_info.capabilities)
            match_ratio = len(required_caps.intersection(node_caps)) / len(required_caps) if required_caps else 1.0
            score += match_ratio * 0.3
            
            # Custom weights
            if node.node_info.node_id in self.node_weights:
                score *= self.node_weights[node.node_info.node_id]
            
            scores[node] = score
        
        return max(scores.keys(), key=lambda n: scores[n])
\`\`\`

### Network Communication Layer
\`\`\`python
import aiohttp
import asyncio
import ssl
from cryptography.fernet import Fernet

class SecureNetworkLayer:
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        self.session = None
        self.message_queue = asyncio.Queue()
        
    async def initialize(self):
        """Initialize network layer"""
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(),
            limit=100,
            limit_per_host=30
        )
        self.session = aiohttp.ClientSession(connector=connector)
    
    async def send_message(self, target_address: str, message: Dict[str, Any], 
                          encrypted: bool = True) -> Dict[str, Any]:
        """Send message to target node"""
        payload = json.dumps(message)
        
        if encrypted:
            payload = self.cipher.encrypt(payload.encode()).decode()
        
        try:
            async with self.session.post(
                f"{target_address}/message",
                json={'payload': payload, 'encrypted': encrypted},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    raise Exception(f"Network error: {response.status}")
        
        except Exception as e:
            raise Exception(f"Failed to send message: {str(e)}")
    
    async def broadcast_message(self, target_addresses: List[str], 
                              message: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast message to multiple nodes"""
        tasks = []
        for address in target_addresses:
            task = asyncio.create_task(self.send_message(address, message))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_responses = []
        failed_responses = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_responses.append({
                    'target': target_addresses[i],
                    'error': str(result)
                })
            else:
                successful_responses.append({
                    'target': target_addresses[i],
                    'response': result
                })
        
        return {
            'successful': successful_responses,
            'failed': failed_responses
        }
    
    def decrypt_message(self, encrypted_payload: str) -> Dict[str, Any]:
        """Decrypt received message"""
        decrypted_data = self.cipher.decrypt(encrypted_payload.encode())
        return json.loads(decrypted_data.decode())

class MessageRouter:
    def __init__(self):
        self.routing_table = {}
        self.message_handlers = {}
        self.middleware = []
    
    def register_route(self, message_type: str, handler: Callable):
        """Register handler for specific message types"""
        self.message_handlers[message_type] = handler
    
    def add_middleware(self, middleware: Callable):
        """Add middleware for message processing"""
        self.middleware.append(middleware)
    
    async def route_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Route message to appropriate handler"""
        message_type = message.get('type', 'unknown')
        
        # Apply middleware
        for middleware in self.middleware:
            message = await middleware(message)
        
        # Route to handler
        if message_type in self.message_handlers:
            handler = self.message_handlers[message_type]
            return await handler(message)
        else:
            return {'error': f'No handler for message type: {message_type}'}
\`\`\`

## Consensus and Coordination

### Distributed Consensus Protocol
\`\`\`python
import random
import hashlib
from enum import Enum

class NodeState(Enum):
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"

class RaftConsensus:
    def __init__(self, node_id: str, cluster_nodes: List[str]):
        self.node_id = node_id
        self.cluster_nodes = cluster_nodes
        self.state = NodeState.FOLLOWER
        
        # Raft state
        self.current_term = 0
        self.voted_for = None
        self.log = []
        self.commit_index = 0
        self.last_applied = 0
        
        # Leader state
        self.next_index = {}
        self.match_index = {}
        
        # Timers
        self.election_timeout = random.uniform(150, 300)  # ms
        self.heartbeat_interval = 50  # ms
        self.last_heartbeat = time.time()
    
    async def start_consensus(self):
        """Start the consensus protocol"""
        while True:
            if self.state == NodeState.LEADER:
                await self._send_heartbeats()
                await asyncio.sleep(self.heartbeat_interval / 1000)
            
            else:
                # Check for election timeout
                if time.time() - self.last_heartbeat > self.election_timeout / 1000:
                    await self._start_election()
                
                await asyncio.sleep(0.1)
    
    async def _start_election(self):
        """Start leader election"""
        self.state = NodeState.CANDIDATE
        self.current_term += 1
        self.voted_for = self.node_id
        self.last_heartbeat = time.time()
        
        # Request votes from other nodes
        votes_received = 1  # Vote for self
        vote_tasks = []
        
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                task = asyncio.create_task(self._request_vote(node_id))
                vote_tasks.append(task)
        
        # Collect votes
        vote_responses = await asyncio.gather(*vote_tasks, return_exceptions=True)
        
        for response in vote_responses:
            if isinstance(response, dict) and response.get('vote_granted', False):
                votes_received += 1
        
        # Check if won election
        majority = (len(self.cluster_nodes) + 1) // 2
        if votes_received > majority:
            await self._become_leader()
        else:
            self.state = NodeState.FOLLOWER
    
    async def _request_vote(self, target_node: str) -> Dict[str, Any]:
        """Request vote from a node"""
        vote_request = {
            'type': 'vote_request',
            'term': self.current_term,
            'candidate_id': self.node_id,
            'last_log_index': len(self.log) - 1,
            'last_log_term': self.log[-1]['term'] if self.log else 0
        }
        
        # Send vote request (implementation depends on network layer)
        response = await self.send_message(target_node, vote_request)
        return response
    
    async def _become_leader(self):
        """Become the leader"""
        self.state = NodeState.LEADER
        
        # Initialize leader state
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                self.next_index[node_id] = len(self.log)
                self.match_index[node_id] = 0
        
        # Send initial heartbeats
        await self._send_heartbeats()
    
    async def _send_heartbeats(self):
        """Send heartbeats to all followers"""
        heartbeat_tasks = []
        
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                task = asyncio.create_task(self._send_append_entries(node_id))
                heartbeat_tasks.append(task)
        
        await asyncio.gather(*heartbeat_tasks, return_exceptions=True)
    
    async def append_log_entry(self, command: Dict[str, Any]) -> bool:
        """Append new log entry (leader only)"""
        if self.state != NodeState.LEADER:
            return False
        
        # Add entry to local log
        log_entry = {
            'term': self.current_term,
            'command': command,
            'index': len(self.log)
        }
        self.log.append(log_entry)
        
        # Replicate to followers
        success_count = 1  # Leader counts as success
        replication_tasks = []
        
        for node_id in self.cluster_nodes:
            if node_id != self.node_id:
                task = asyncio.create_task(self._replicate_entry(node_id, log_entry))
                replication_tasks.append(task)
        
        # Wait for majority replication
        responses = await asyncio.gather(*replication_tasks, return_exceptions=True)
        
        for response in responses:
            if isinstance(response, dict) and response.get('success', False):
                success_count += 1
        
        # Commit if majority successful
        majority = (len(self.cluster_nodes) + 1) // 2
        if success_count > majority:
            self.commit_index = log_entry['index']
            return True
        
        return False

class DistributedLock:
    def __init__(self, lock_id: str, consensus_system: RaftConsensus):
        self.lock_id = lock_id
        self.consensus = consensus_system
        self.holder = None
        self.lease_expiry = None
        self.lease_duration = 30  # seconds
    
    async def acquire(self, requester_id: str) -> bool:
        """Acquire distributed lock"""
        current_time = time.time()
        
        # Check if lock is available
        if self.holder and self.lease_expiry > current_time:
            return False  # Lock is held
        
        # Attempt to acquire through consensus
        acquire_command = {
            'type': 'acquire_lock',
            'lock_id': self.lock_id,
            'requester_id': requester_id,
            'timestamp': current_time
        }
        
        success = await self.consensus.append_log_entry(acquire_command)
        
        if success:
            self.holder = requester_id
            self.lease_expiry = current_time + self.lease_duration
            return True
        
        return False
    
    async def release(self, requester_id: str) -> bool:
        """Release distributed lock"""
        if self.holder != requester_id:
            return False  # Not the lock holder
        
        release_command = {
            'type': 'release_lock',
            'lock_id': self.lock_id,
            'requester_id': requester_id
        }
        
        success = await self.consensus.append_log_entry(release_command)
        
        if success:
            self.holder = None
            self.lease_expiry = None
            return True
        
        return False
\`\`\`

## Fault Tolerance and Recovery

### Circuit Breaker Pattern
\`\`\`python
from enum import Enum
import statistics

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, 
                 recovery_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_threshold = recovery_threshold
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.call_history = []
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.recovery_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.success_count = 0
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset"""
        return (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout)

class FaultTolerantCluster:
    def __init__(self, nodes: List[DistributedNode]):
        self.nodes = {node.node_info.node_id: node for node in nodes}
        self.circuit_breakers = {node_id: CircuitBreaker() for node_id in self.nodes}
        self.health_monitor = HealthMonitor(self.nodes)
        
    async def execute_task_with_failover(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task with automatic failover"""
        required_capabilities = task.get('required_capabilities', [])
        
        # Get eligible nodes
        eligible_nodes = [
            node for node in self.nodes.values()
            if all(cap in node.node_info.capabilities for cap in required_capabilities)
        ]
        
        # Sort by health and load
        eligible_nodes.sort(key=lambda n: (n.node_info.current_load, 
                                          1 if n.node_info.health_status == "healthy" else 2))
        
        last_exception = None
        
        for node in eligible_nodes:
            try:
                circuit_breaker = self.circuit_breakers[node.node_info.node_id]
                result = await circuit_breaker.call(node.process_task, task)
                return result
            
            except Exception as e:
                last_exception = e
                continue
        
        # All nodes failed
        raise Exception(f"All eligible nodes failed. Last error: {str(last_exception)}")
    
    async def start_cluster(self):
        """Start the fault-tolerant cluster"""
        # Start all nodes
        node_tasks = [asyncio.create_task(node.start()) for node in self.nodes.values()]
        
        # Start health monitoring
        health_task = asyncio.create_task(self.health_monitor.start_monitoring())
        
        # Wait for all tasks
        await asyncio.gather(*node_tasks, health_task)

class HealthMonitor:
    def __init__(self, nodes: Dict[str, DistributedNode]):
        self.nodes = nodes
        self.health_history = {node_id: [] for node_id in nodes}
        self.monitoring_interval = 30  # seconds
    
    async def start_monitoring(self):
        """Start health monitoring for all nodes"""
        while True:
            await self._check_all_nodes()
            await asyncio.sleep(self.monitoring_interval)
    
    async def _check_all_nodes(self):
        """Check health of all nodes"""
        check_tasks = []
        
        for node_id, node in self.nodes.items():
            task = asyncio.create_task(self._check_node_health(node_id, node))
            check_tasks.append(task)
        
        await asyncio.gather(*check_tasks, return_exceptions=True)
    
    async def _check_node_health(self, node_id: str, node: DistributedNode):
        """Check health of a specific node"""
        try:
            # Ping node
            start_time = time.time()
            health_check = {'type': 'health_check', 'timestamp': start_time}
            
            response = await asyncio.wait_for(
                node.process_task(health_check),
                timeout=10.0
            )
            
            response_time = time.time() - start_time
            
            # Update health metrics
            health_data = {
                'timestamp': start_time,
                'response_time': response_time,
                'status': 'healthy'
            }
            
            self.health_history[node_id].append(health_data)
            node.node_info.health_status = "healthy"
        
        except Exception as e:
            # Node is unhealthy
            health_data = {
                'timestamp': time.time(),
                'error': str(e),
                'status': 'unhealthy'
            }
            
            self.health_history[node_id].append(health_data)
            node.node_info.health_status = "offline"
        
        # Keep only recent history
        self.health_history[node_id] = self.health_history[node_id][-100:]
\`\`\`

Distributed agentic AI systems require careful design of consensus protocols, fault tolerance mechanisms, and communication layers to ensure reliable operation across multiple nodes while maintaining system coherence and performance.`,
    date: "2024-01-28",
    tags: ["Distributed Systems", "Agentic AI", "Fault Tolerance", "Consensus"],
    readTime: "26 min read",
    category: "Agentic AI"
  },
  {
    id: "mcp-client-development",
    title: "MCP Client Development: Building Efficient AI Tool Integration",
    description: "Complete guide to developing Model Context Protocol (MCP) clients for seamless integration between AI models and external tools and services.",
    content: `# MCP Client Development: Building Efficient AI Tool Integration

The Model Context Protocol (MCP) enables AI models to interact with external tools and services through a standardized interface, creating more capable and versatile AI applications.

## MCP Protocol Fundamentals

### Understanding the MCP Architecture
\`\`\`python
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import asyncio
import json
import websockets

@dataclass
class MCPMessage:
    """Base MCP message structure"""
    id: str
    method: str
    params: Dict[str, Any]
    jsonrpc: str = "2.0"

@dataclass
class MCPResponse:
    """MCP response structure"""
    id: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    jsonrpc: str = "2.0"

@dataclass
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None

class MCPClient:
    """MCP client for connecting to MCP servers"""
    
    def __init__(self, server_uri: str):
        self.server_uri = server_uri
        self.websocket = None
        self.tools = {}
        self.message_id_counter = 0
        self.pending_requests = {}
        
    async def connect(self):
        """Connect to MCP server"""
        self.websocket = await websockets.connect(self.server_uri)
        await self._initialize_session()
        
        # Start message handler
        asyncio.create_task(self._handle_messages())
    
    async def _initialize_session(self):
        """Initialize MCP session"""
        init_message = MCPMessage(
            id=self._next_message_id(),
            method="initialize",
            params={
                "protocolVersion": "1.0",
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"subscribe": True, "listChanged": True}
                },
                "clientInfo": {
                    "name": "python-mcp-client",
                    "version": "1.0.0"
                }
            }
        )
        
        response = await self._send_request(init_message)
        if response.error:
            raise Exception(f"Initialization failed: {response.error}")
        
        # List available tools
        await self._list_tools()
    
    async def _list_tools(self):
        """List available tools from server"""
        list_message = MCPMessage(
            id=self._next_message_id(),
            method="tools/list",
            params={}
        )
        
        response = await self._send_request(list_message)
        if response.result and 'tools' in response.result:
            for tool_def in response.result['tools']:
                tool = MCPTool(
                    name=tool_def['name'],
                    description=tool_def['description'],
                    input_schema=tool_def['inputSchema']
                )
                self.tools[tool.name] = tool
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        # Validate arguments against schema
        tool = self.tools[tool_name]
        self._validate_arguments(arguments, tool.input_schema)
        
        call_message = MCPMessage(
            id=self._next_message_id(),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments
            }
        )
        
        response = await self._send_request(call_message)
        
        if response.error:
            raise Exception(f"Tool call failed: {response.error}")
        
        return response.result
    
    async def _send_request(self, message: MCPMessage) -> MCPResponse:
        """Send request and wait for response"""
        future = asyncio.Future()
        self.pending_requests[message.id] = future
        
        await self.websocket.send(json.dumps(asdict(message)))
        
        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        finally:
            self.pending_requests.pop(message.id, None)
    
    async def _handle_messages(self):
        """Handle incoming messages from server"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                
                if 'id' in data and data['id'] in self.pending_requests:
                    # Response to pending request
                    future = self.pending_requests[data['id']]
                    response = MCPResponse(
                        id=data['id'],
                        result=data.get('result'),
                        error=data.get('error')
                    )
                    future.set_result(response)
                
                elif 'method' in data:
                    # Notification from server
                    await self._handle_notification(data)
        
        except websockets.exceptions.ConnectionClosed:
            print("Connection to MCP server closed")
    
    async def _handle_notification(self, notification: Dict[str, Any]):
        """Handle notifications from server"""
        method = notification['method']
        
        if method == "notifications/tools/list_changed":
            # Tool list changed, refresh
            await self._list_tools()
        
        elif method == "notifications/progress":
            # Progress update
            progress_data = notification['params']
            print(f"Progress: {progress_data}")
    
    def _validate_arguments(self, arguments: Dict[str, Any], schema: Dict[str, Any]):
        """Validate arguments against JSON schema"""
        # Simple validation - in production, use jsonschema library
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in arguments:
                raise ValueError(f"Required field '{field}' missing")
    
    def _next_message_id(self) -> str:
        """Generate next message ID"""
        self.message_id_counter += 1
        return str(self.message_id_counter)
\`\`\`

### Advanced MCP Client Features
\`\`\`python
import asyncio
import time
from typing import Callable, Optional
import logging

class AdvancedMCPClient(MCPClient):
    """Enhanced MCP client with additional features"""
    
    def __init__(self, server_uri: str, max_retries: int = 3, 
                 connection_timeout: float = 10.0):
        super().__init__(server_uri)
        self.max_retries = max_retries
        self.connection_timeout = connection_timeout
        self.retry_delay = 1.0
        self.logger = logging.getLogger(__name__)
        
        # Tool caching
        self.tool_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Rate limiting
        self.rate_limiter = RateLimiter(requests_per_second=10)
        
        # Connection health
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 30.0
        
        # Middleware
        self.middleware = []
    
    async def connect_with_retry(self):
        """Connect with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await asyncio.wait_for(self.connect(), timeout=self.connection_timeout)
                self.logger.info("Successfully connected to MCP server")
                
                # Start heartbeat monitoring
                asyncio.create_task(self._heartbeat_monitor())
                return
                
            except Exception as e:
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    raise Exception(f"Failed to connect after {self.max_retries} attempts")
    
    async def call_tool_with_caching(self, tool_name: str, arguments: Dict[str, Any], 
                                   cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Call tool with result caching"""
        if cache_key is None:
            cache_key = f"{tool_name}:{hash(json.dumps(arguments, sort_keys=True))}"
        
        # Check cache
        if cache_key in self.tool_cache:
            cached_result, timestamp = self.tool_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                self.logger.debug(f"Cache hit for {tool_name}")
                return cached_result
        
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Apply middleware
        request_data = {'tool_name': tool_name, 'arguments': arguments}
        for middleware in self.middleware:
            request_data = await middleware.before_request(request_data)
        
        try:
            # Call tool
            result = await self.call_tool(request_data['tool_name'], request_data['arguments'])
            
            # Cache result
            self.tool_cache[cache_key] = (result, time.time())
            
            # Apply response middleware
            for middleware in self.middleware:
                result = await middleware.after_response(request_data, result)
            
            return result
            
        except Exception as e:
            # Apply error middleware
            for middleware in self.middleware:
                await middleware.on_error(request_data, e)
            raise
    
    async def batch_tool_calls(self, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute multiple tool calls in parallel"""
        tasks = []
        
        for call in calls:
            task = asyncio.create_task(
                self.call_tool_with_caching(
                    call['tool_name'], 
                    call['arguments'],
                    call.get('cache_key')
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'call': calls[i]
                })
            else:
                processed_results.append({
                    'success': True,
                    'result': result,
                    'call': calls[i]
                })
        
        return processed_results
    
    async def _heartbeat_monitor(self):
        """Monitor connection health with heartbeats"""
        while self.websocket and not self.websocket.closed:
            try:
                # Send ping
                ping_message = MCPMessage(
                    id=self._next_message_id(),
                    method="ping",
                    params={}
                )
                
                start_time = time.time()
                response = await self._send_request(ping_message)
                latency = time.time() - start_time
                
                if response.error:
                    self.logger.warning(f"Heartbeat failed: {response.error}")
                else:
                    self.last_heartbeat = time.time()
                    self.logger.debug(f"Heartbeat successful, latency: {latency:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                # Attempt reconnection
                await self.connect_with_retry()
            
            await asyncio.sleep(self.heartbeat_interval)

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, requests_per_second: float):
        self.requests_per_second = requests_per_second
        self.bucket_capacity = requests_per_second
        self.tokens = requests_per_second
        self.last_refill = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_refill
            
            # Refill tokens
            self.tokens = min(
                self.bucket_capacity,
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_refill = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return
            
            # Wait for next token
            wait_time = (1 - self.tokens) / self.requests_per_second
            await asyncio.sleep(wait_time)
            self.tokens = 0

class MCPMiddleware(ABC):
    """Base class for MCP middleware"""
    
    @abstractmethod
    async def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process request before sending"""
        return request_data
    
    @abstractmethod
    async def after_response(self, request_data: Dict[str, Any], 
                           response: Dict[str, Any]) -> Dict[str, Any]:
        """Process response after receiving"""
        return response
    
    @abstractmethod
    async def on_error(self, request_data: Dict[str, Any], error: Exception):
        """Handle errors"""
        pass

class LoggingMiddleware(MCPMiddleware):
    """Middleware for logging requests and responses"""
    
    def __init__(self):
        self.logger = logging.getLogger("mcp.middleware.logging")
    
    async def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Calling tool: {request_data['tool_name']}")
        self.logger.debug(f"Arguments: {request_data['arguments']}")
        return request_data
    
    async def after_response(self, request_data: Dict[str, Any], 
                           response: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Tool {request_data['tool_name']} completed successfully")
        return response
    
    async def on_error(self, request_data: Dict[str, Any], error: Exception):
        self.logger.error(f"Tool {request_data['tool_name']} failed: {error}")

class AuthenticationMiddleware(MCPMiddleware):
    """Middleware for adding authentication"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def before_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        # Add authentication to arguments
        if 'headers' not in request_data['arguments']:
            request_data['arguments']['headers'] = {}
        request_data['arguments']['headers']['Authorization'] = f"Bearer {self.api_key}"
        return request_data
    
    async def after_response(self, request_data: Dict[str, Any], 
                           response: Dict[str, Any]) -> Dict[str, Any]:
        return response
    
    async def on_error(self, request_data: Dict[str, Any], error: Exception):
        if "unauthorized" in str(error).lower():
            logging.warning("Authentication may have failed")
\`\`\`

## Tool Integration Patterns

### File System Tools
\`\`\`python
import os
import shutil
from pathlib import Path

class FileSystemMCPTools:
    """MCP tools for file system operations"""
    
    @staticmethod
    async def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """Read file contents"""
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": f"File not found: {file_path}"}
            
            if not path.is_file():
                return {"error": f"Not a file: {file_path}"}
            
            content = path.read_text(encoding=encoding)
            return {
                "content": content,
                "size": len(content),
                "encoding": encoding,
                "path": str(path.absolute())
            }
            
        except Exception as e:
            return {"error": f"Failed to read file: {str(e)}"}
    
    @staticmethod
    async def write_file(file_path: str, content: str, 
                        encoding: str = "utf-8", create_dirs: bool = True) -> Dict[str, Any]:
        """Write content to file"""
        try:
            path = Path(file_path)
            
            if create_dirs:
                path.parent.mkdir(parents=True, exist_ok=True)
            
            path.write_text(content, encoding=encoding)
            
            return {
                "success": True,
                "path": str(path.absolute()),
                "size": len(content)
            }
            
        except Exception as e:
            return {"error": f"Failed to write file: {str(e)}"}
    
    @staticmethod
    async def list_directory(directory_path: str, recursive: bool = False, 
                           include_hidden: bool = False) -> Dict[str, Any]:
        """List directory contents"""
        try:
            path = Path(directory_path)
            if not path.exists():
                return {"error": f"Directory not found: {directory_path}"}
            
            if not path.is_dir():
                return {"error": f"Not a directory: {directory_path}"}
            
            items = []
            
            if recursive:
                pattern = "**/*" if include_hidden else "**/[!.]*"
                for item in path.glob(pattern):
                    items.append({
                        "name": item.name,
                        "path": str(item.absolute()),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None
                    })
            else:
                for item in path.iterdir():
                    if not include_hidden and item.name.startswith('.'):
                        continue
                    
                    items.append({
                        "name": item.name,
                        "path": str(item.absolute()),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None
                    })
            
            return {
                "items": items,
                "count": len(items),
                "directory": str(path.absolute())
            }
            
        except Exception as e:
            return {"error": f"Failed to list directory: {str(e)}"}

class DatabaseMCPTools:
    """MCP tools for database operations"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    async def connect(self):
        """Connect to database"""
        # Implementation depends on database type
        pass
    
    async def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute database query"""
        try:
            if not self.connection:
                await self.connect()
            
            # Execute query with parameters
            cursor = await self.connection.execute(query, parameters or {})
            
            if query.strip().upper().startswith('SELECT'):
                # Fetch results for SELECT queries
                rows = await cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                
                results = []
                for row in rows:
                    results.append(dict(zip(columns, row)))
                
                return {
                    "results": results,
                    "row_count": len(results),
                    "columns": columns
                }
            else:
                # For INSERT, UPDATE, DELETE
                row_count = cursor.rowcount
                await self.connection.commit()
                
                return {
                    "success": True,
                    "affected_rows": row_count
                }
                
        except Exception as e:
            return {"error": f"Database query failed: {str(e)}"}
    
    async def get_schema_info(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get database schema information"""
        try:
            if table_name:
                # Get specific table schema
                query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = :table_name
                ORDER BY ordinal_position
                """
                result = await self.execute_query(query, {"table_name": table_name})
                return result
            else:
                # Get all tables
                query = """
                SELECT table_name, table_type 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
                """
                result = await self.execute_query(query)
                return result
                
        except Exception as e:
            return {"error": f"Failed to get schema info: {str(e)}"}

class WebScrapingMCPTools:
    """MCP tools for web scraping"""
    
    def __init__(self):
        self.session = None
    
    async def fetch_webpage(self, url: str, headers: Optional[Dict[str, str]] = None,
                           timeout: float = 30.0) -> Dict[str, Any]:
        """Fetch webpage content"""
        try:
            import aiohttp
            from bs4 import BeautifulSoup
            
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            async with self.session.get(url, headers=headers, timeout=timeout) as response:
                content = await response.text()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extract metadata
                title = soup.find('title')
                meta_description = soup.find('meta', attrs={'name': 'description'})
                
                return {
                    "url": url,
                    "status_code": response.status,
                    "title": title.text.strip() if title else None,
                    "description": meta_description.get('content') if meta_description else None,
                    "content": content,
                    "text_content": soup.get_text(strip=True),
                    "links": [{"text": a.text.strip(), "href": a.get('href')} 
                             for a in soup.find_all('a', href=True)],
                    "images": [{"alt": img.get('alt', ''), "src": img.get('src')} 
                              for img in soup.find_all('img', src=True)]
                }
                
        except Exception as e:
            return {"error": f"Failed to fetch webpage: {str(e)}"}
    
    async def extract_structured_data(self, url: str, selectors: Dict[str, str]) -> Dict[str, Any]:
        """Extract structured data using CSS selectors"""
        try:
            webpage_result = await self.fetch_webpage(url)
            
            if "error" in webpage_result:
                return webpage_result
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(webpage_result["content"], 'html.parser')
            
            extracted_data = {}
            
            for field_name, selector in selectors.items():
                elements = soup.select(selector)
                
                if len(elements) == 1:
                    extracted_data[field_name] = elements[0].text.strip()
                elif len(elements) > 1:
                    extracted_data[field_name] = [el.text.strip() for el in elements]
                else:
                    extracted_data[field_name] = None
            
            return {
                "url": url,
                "extracted_data": extracted_data,
                "extraction_count": len([v for v in extracted_data.values() if v is not None])
            }
            
        except Exception as e:
            return {"error": f"Failed to extract structured data: {str(e)}"}
\`\`\`

## Production Deployment

### MCP Client Manager
\`\`\`python
class MCPClientManager:
    """Production-ready MCP client manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.clients = {}
        self.health_monitor = MCPHealthMonitor()
        self.metrics_collector = MCPMetricsCollector()
        
    async def initialize_clients(self):
        """Initialize all configured MCP clients"""
        for client_config in self.config['clients']:
            client_id = client_config['id']
            
            client = AdvancedMCPClient(
                server_uri=client_config['server_uri'],
                max_retries=client_config.get('max_retries', 3),
                connection_timeout=client_config.get('connection_timeout', 10.0)
            )
            
            # Add middleware
            if client_config.get('logging', True):
                client.middleware.append(LoggingMiddleware())
            
            if 'api_key' in client_config:
                client.middleware.append(AuthenticationMiddleware(client_config['api_key']))
            
            # Connect client
            await client.connect_with_retry()
            
            # Register with health monitor
            self.health_monitor.register_client(client_id, client)
            
            self.clients[client_id] = client
    
    async def get_client(self, client_id: str) -> Optional[AdvancedMCPClient]:
        """Get client by ID"""
        return self.clients.get(client_id)
    
    async def call_tool_on_any_client(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call tool on any available client that supports it"""
        for client_id, client in self.clients.items():
            if tool_name in client.tools:
                try:
                    result = await client.call_tool_with_caching(tool_name, arguments)
                    await self.metrics_collector.record_success(client_id, tool_name)
                    return result
                except Exception as e:
                    await self.metrics_collector.record_error(client_id, tool_name, str(e))
                    continue
        
        raise Exception(f"No available client supports tool: {tool_name}")
    
    async def start_health_monitoring(self):
        """Start health monitoring for all clients"""
        await self.health_monitor.start_monitoring()
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return await self.metrics_collector.get_metrics()

class MCPHealthMonitor:
    """Health monitoring for MCP clients"""
    
    def __init__(self):
        self.clients = {}
        self.health_status = {}
        self.monitoring_interval = 60  # seconds
    
    def register_client(self, client_id: str, client: AdvancedMCPClient):
        """Register client for monitoring"""
        self.clients[client_id] = client
        self.health_status[client_id] = {
            'status': 'unknown',
            'last_check': None,
            'consecutive_failures': 0
        }
    
    async def start_monitoring(self):
        """Start monitoring all registered clients"""
        while True:
            await self._check_all_clients()
            await asyncio.sleep(self.monitoring_interval)
    
    async def _check_all_clients(self):
        """Check health of all clients"""
        for client_id, client in self.clients.items():
            try:
                # Simple health check
                start_time = time.time()
                await client.call_tool("ping", {})
                response_time = time.time() - start_time
                
                self.health_status[client_id] = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'response_time': response_time,
                    'consecutive_failures': 0
                }
                
            except Exception as e:
                self.health_status[client_id]['consecutive_failures'] += 1
                self.health_status[client_id]['status'] = 'unhealthy'
                self.health_status[client_id]['last_error'] = str(e)
                
                # Attempt reconnection after multiple failures
                if self.health_status[client_id]['consecutive_failures'] >= 3:
                    await self._attempt_reconnection(client_id, client)
    
    async def _attempt_reconnection(self, client_id: str, client: AdvancedMCPClient):
        """Attempt to reconnect a failed client"""
        try:
            await client.connect_with_retry()
            self.health_status[client_id]['consecutive_failures'] = 0
            self.health_status[client_id]['status'] = 'recovered'
        except Exception as e:
            logging.error(f"Failed to reconnect client {client_id}: {e}")

class MCPMetricsCollector:
    """Metrics collection for MCP operations"""
    
    def __init__(self):
        self.metrics = {
            'tool_calls': {},
            'errors': {},
            'response_times': {},
            'success_rate': {}
        }
    
    async def record_success(self, client_id: str, tool_name: str, response_time: float = 0):
        """Record successful tool call"""
        key = f"{client_id}:{tool_name}"
        
        if key not in self.metrics['tool_calls']:
            self.metrics['tool_calls'][key] = 0
        self.metrics['tool_calls'][key] += 1
        
        if key not in self.metrics['response_times']:
            self.metrics['response_times'][key] = []
        self.metrics['response_times'][key].append(response_time)
        
        # Update success rate
        self._update_success_rate(key, True)
    
    async def record_error(self, client_id: str, tool_name: str, error: str):
        """Record tool call error"""
        key = f"{client_id}:{tool_name}"
        
        if key not in self.metrics['errors']:
            self.metrics['errors'][key] = []
        self.metrics['errors'][key].append({
            'error': error,
            'timestamp': time.time()
        })
        
        # Update success rate
        self._update_success_rate(key, False)
    
    def _update_success_rate(self, key: str, success: bool):
        """Update success rate for a tool"""
        if key not in self.metrics['success_rate']:
            self.metrics['success_rate'][key] = {'total': 0, 'successful': 0}
        
        self.metrics['success_rate'][key]['total'] += 1
        if success:
            self.metrics['success_rate'][key]['successful'] += 1
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            'tool_calls': self.metrics['tool_calls'],
            'error_count': {k: len(v) for k, v in self.metrics['errors'].items()},
            'average_response_times': {
                k: sum(v) / len(v) if v else 0 
                for k, v in self.metrics['response_times'].items()
            },
            'success_rates': {
                k: v['successful'] / v['total'] if v['total'] > 0 else 0
                for k, v in self.metrics['success_rate'].items()
            }
        }
\`\`\`

MCP client development enables powerful AI-tool integration through standardized protocols, robust error handling, and comprehensive monitoring systems that ensure reliable operation in production environments.`,
    date: "2024-01-25",
    tags: ["MCP", "Tool Integration", "AI Infrastructure", "Client Development"],
    readTime: "20 min read",
    category: "MCP"
  },
  {
    id: "mcp-server-implementation",
    title: "MCP Server Implementation: Creating Robust AI Tool Backends",
    description: "Detailed guide to implementing Model Context Protocol (MCP) servers that provide reliable, scalable tool interfaces for AI applications.",
    content: `# MCP Server Implementation: Creating Robust AI Tool Backends

MCP servers act as the backbone of AI tool ecosystems, providing standardized interfaces for AI models to interact with external services, databases, and APIs.

## MCP Server Architecture

### Core Server Implementation
\`\`\`python
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import asyncio
import json
import websockets
import logging
from abc import ABC, abstractmethod

@dataclass
class MCPToolDefinition:
    """Tool definition for MCP server"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None
    handler: Optional[Callable] = None

@dataclass
class MCPResource:
    """Resource definition for MCP server"""
    uri: str
    name: str
    description: str
    mime_type: str

class MCPServerBase(ABC):
    """Base class for MCP servers"""
    
    def __init__(self, server_name: str, version: str = "1.0.0"):
        self.server_name = server_name
        self.version = version
        self.tools = {}
        self.resources = {}
        self.prompts = {}
        self.connected_clients = set()
        self.logger = logging.getLogger(__name__)
        
        # Server capabilities
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "prompts": {"listChanged": True}
        }
    
    def register_tool(self, tool: MCPToolDefinition):
        """Register a tool with the server"""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
    
    def register_resource(self, resource: MCPResource):
        """Register a resource with the server"""
        self.resources[resource.uri] = resource
        self.logger.info(f"Registered resource: {resource.uri}")
    
    async def start_server(self, host: str = "localhost", port: int = 8765):
        """Start the MCP server"""
        self.logger.info(f"Starting MCP server on {host}:{port}")
        
        async def client_handler(websocket, path):
            await self._handle_client_connection(websocket)
        
        await websockets.serve(client_handler, host, port)
        self.logger.info("MCP server started successfully")
    
    async def _handle_client_connection(self, websocket):
        """Handle a client connection"""
        client_id = id(websocket)
        self.connected_clients.add(client_id)
        self.logger.info(f"Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self._process_message(data)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        finally:
            self.connected_clients.discard(client_id)
    
    async def _process_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process incoming message from client"""
        method = data.get("method")
        message_id = data.get("id")
        params = data.get("params", {})
        
        try:
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_list_tools(params)
            elif method == "tools/call":
                result = await self._handle_call_tool(params)
            elif method == "resources/list":
                result = await self._handle_list_resources(params)
            elif method == "resources/read":
                result = await self._handle_read_resource(params)
            elif method == "prompts/list":
                result = await self._handle_list_prompts(params)
            elif method == "prompts/get":
                result = await self._handle_get_prompt(params)
            else:
                raise Exception(f"Unknown method: {method}")
            
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Error processing method {method}: {e}")
            return {
                "jsonrpc": "2.0",
                "id": message_id,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
    
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialization request"""
        return {
            "protocolVersion": "1.0",
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.server_name,
                "version": self.version
            }
        }
    
    async def _handle_list_tools(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools list request"""
        tools_list = []
        
        for tool_name, tool in self.tools.items():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema
            })
        
        return {"tools": tools_list}
    
    async def _handle_call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call request"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            raise Exception(f"Tool not found: {tool_name}")
        
        tool = self.tools[tool_name]
        
        # Validate arguments
        self._validate_tool_arguments(arguments, tool.input_schema)
        
        # Execute tool
        if tool.handler:
            result = await tool.handler(arguments)
        else:
            result = await self._execute_tool(tool_name, arguments)
        
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result) if isinstance(result, dict) else str(result)
                }
            ]
        }
    
    def _validate_tool_arguments(self, arguments: Dict[str, Any], schema: Dict[str, Any]):
        """Validate tool arguments against schema"""
        required_fields = schema.get("required", [])
        
        for field in required_fields:
            if field not in arguments:
                raise Exception(f"Required field missing: {field}")
        
        # Additional validation can be added here
    
    @abstractmethod
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute tool - to be implemented by subclasses"""
        pass

class FileSystemMCPServer(MCPServerBase):
    """MCP server for file system operations"""
    
    def __init__(self, allowed_paths: List[str] = None):
        super().__init__("filesystem-server", "1.0.0")
        self.allowed_paths = allowed_paths or ["/tmp"]
        self._register_filesystem_tools()
    
    def _register_filesystem_tools(self):
        """Register file system tools"""
        # Read file tool
        read_file_tool = MCPToolDefinition(
            name="read_file",
            description="Read contents of a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path"]
            }
        )
        self.register_tool(read_file_tool)
        
        # Write file tool
        write_file_tool = MCPToolDefinition(
            name="write_file",
            description="Write content to a file",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "encoding": {"type": "string", "default": "utf-8"}
                },
                "required": ["path", "content"]
            }
        )
        self.register_tool(write_file_tool)
        
        # List directory tool
        list_dir_tool = MCPToolDefinition(
            name="list_directory",
            description="List contents of a directory",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list"},
                    "recursive": {"type": "boolean", "default": False}
                },
                "required": ["path"]
            }
        )
        self.register_tool(list_dir_tool)
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if path is within allowed directories"""
        import os
        abs_path = os.path.abspath(path)
        
        for allowed_path in self.allowed_paths:
            if abs_path.startswith(os.path.abspath(allowed_path)):
                return True
        
        return False
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute file system tool"""
        import os
        from pathlib import Path
        
        if tool_name == "read_file":
            path = arguments["path"]
            encoding = arguments.get("encoding", "utf-8")
            
            if not self._is_path_allowed(path):
                raise Exception(f"Access denied to path: {path}")
            
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                return {
                    "content": content,
                    "path": path,
                    "size": len(content)
                }
            except Exception as e:
                raise Exception(f"Failed to read file: {str(e)}")
        
        elif tool_name == "write_file":
            path = arguments["path"]
            content = arguments["content"]
            encoding = arguments.get("encoding", "utf-8")
            
            if not self._is_path_allowed(path):
                raise Exception(f"Access denied to path: {path}")
            
            try:
                # Create directory if it doesn't exist
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "path": path,
                    "bytes_written": len(content.encode(encoding))
                }
            except Exception as e:
                raise Exception(f"Failed to write file: {str(e)}")
        
        elif tool_name == "list_directory":
            path = arguments["path"]
            recursive = arguments.get("recursive", False)
            
            if not self._is_path_allowed(path):
                raise Exception(f"Access denied to path: {path}")
            
            try:
                path_obj = Path(path)
                if not path_obj.exists():
                    raise Exception(f"Directory does not exist: {path}")
                
                if not path_obj.is_dir():
                    raise Exception(f"Path is not a directory: {path}")
                
                items = []
                
                if recursive:
                    for item in path_obj.rglob("*"):
                        if self._is_path_allowed(str(item)):
                            items.append({
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if item.is_dir() else "file",
                                "size": item.stat().st_size if item.is_file() else None
                            })
                else:
                    for item in path_obj.iterdir():
                        if self._is_path_allowed(str(item)):
                            items.append({
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if item.is_dir() else "file",
                                "size": item.stat().st_size if item.is_file() else None
                            })
                
                return {
                    "items": items,
                    "count": len(items),
                    "path": path
                }
            except Exception as e:
                raise Exception(f"Failed to list directory: {str(e)}")
        
        else:
            raise Exception(f"Unknown tool: {tool_name}")
\`\`\`

### Database MCP Server
\`\`\`python
import asyncpg
import sqlite3
import aiosqlite
from typing import Union

class DatabaseMCPServer(MCPServerBase):
    """MCP server for database operations"""
    
    def __init__(self, database_config: Dict[str, Any]):
        super().__init__("database-server", "1.0.0")
        self.database_config = database_config
        self.connection_pool = None
        self._register_database_tools()
    
    def _register_database_tools(self):
        """Register database tools"""
        # Execute query tool
        query_tool = MCPToolDefinition(
            name="execute_query",
            description="Execute a SQL query",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query to execute"},
                    "parameters": {
                        "type": "object",
                        "description": "Query parameters",
                        "default": {}
                    }
                },
                "required": ["query"]
            }
        )
        self.register_tool(query_tool)
        
        # Get schema tool
        schema_tool = MCPToolDefinition(
            name="get_schema",
            description="Get database schema information",
            input_schema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Specific table name (optional)"
                    }
                }
            }
        )
        self.register_tool(schema_tool)
        
        # List tables tool
        tables_tool = MCPToolDefinition(
            name="list_tables",
            description="List all tables in the database",
            input_schema={
                "type": "object",
                "properties": {}
            }
        )
        self.register_tool(tables_tool)
    
    async def initialize_database(self):
        """Initialize database connection"""
        db_type = self.database_config.get("type", "sqlite")
        
        if db_type == "postgresql":
            self.connection_pool = await asyncpg.create_pool(
                host=self.database_config["host"],
                port=self.database_config.get("port", 5432),
                user=self.database_config["user"],
                password=self.database_config["password"],
                database=self.database_config["database"],
                min_size=1,
                max_size=10
            )
        elif db_type == "sqlite":
            # SQLite doesn't use connection pools in the same way
            self.database_path = self.database_config["path"]
        else:
            raise Exception(f"Unsupported database type: {db_type}")
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute database tool"""
        db_type = self.database_config.get("type", "sqlite")
        
        if tool_name == "execute_query":
            query = arguments["query"]
            parameters = arguments.get("parameters", {})
            
            if db_type == "postgresql":
                return await self._execute_postgresql_query(query, parameters)
            elif db_type == "sqlite":
                return await self._execute_sqlite_query(query, parameters)
        
        elif tool_name == "get_schema":
            table_name = arguments.get("table_name")
            
            if db_type == "postgresql":
                return await self._get_postgresql_schema(table_name)
            elif db_type == "sqlite":
                return await self._get_sqlite_schema(table_name)
        
        elif tool_name == "list_tables":
            if db_type == "postgresql":
                return await self._list_postgresql_tables()
            elif db_type == "sqlite":
                return await self._list_sqlite_tables()
        
        else:
            raise Exception(f"Unknown tool: {tool_name}")
    
    async def _execute_postgresql_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute PostgreSQL query"""
        async with self.connection_pool.acquire() as connection:
            try:
                if query.strip().upper().startswith('SELECT'):
                    # SELECT query
                    rows = await connection.fetch(query, *parameters.values())
                    
                    results = []
                    for row in rows:
                        results.append(dict(row))
                    
                    return {
                        "results": results,
                        "row_count": len(results),
                        "columns": list(rows[0].keys()) if rows else []
                    }
                else:
                    # INSERT, UPDATE, DELETE
                    result = await connection.execute(query, *parameters.values())
                    
                    # Extract affected row count from result
                    affected_rows = int(result.split()[-1]) if result else 0
                    
                    return {
                        "success": True,
                        "affected_rows": affected_rows,
                        "message": result
                    }
                    
            except Exception as e:
                raise Exception(f"Database error: {str(e)}")
    
    async def _execute_sqlite_query(self, query: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SQLite query"""
        async with aiosqlite.connect(self.database_path) as connection:
            try:
                cursor = await connection.execute(query, list(parameters.values()))
                
                if query.strip().upper().startswith('SELECT'):
                    # SELECT query
                    rows = await cursor.fetchall()
                    
                    # Get column names
                    columns = [description[0] for description in cursor.description]
                    
                    results = []
                    for row in rows:
                        results.append(dict(zip(columns, row)))
                    
                    return {
                        "results": results,
                        "row_count": len(results),
                        "columns": columns
                    }
                else:
                    # INSERT, UPDATE, DELETE
                    await connection.commit()
                    
                    return {
                        "success": True,
                        "affected_rows": cursor.rowcount,
                        "last_row_id": cursor.lastrowid
                    }
                    
            except Exception as e:
                raise Exception(f"Database error: {str(e)}")
    
    async def _get_postgresql_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get PostgreSQL schema information"""
        if table_name:
            query = """
            SELECT column_name, data_type, is_nullable, column_default, ordinal_position
            FROM information_schema.columns 
            WHERE table_name = $1 AND table_schema = 'public'
            ORDER BY ordinal_position
            """
            return await self._execute_postgresql_query(query, {"table_name": table_name})
        else:
            query = """
            SELECT table_name, table_type 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
            return await self._execute_postgresql_query(query, {})
    
    async def _get_sqlite_schema(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Get SQLite schema information"""
        if table_name:
            query = f"PRAGMA table_info({table_name})"
            return await self._execute_sqlite_query(query, {})
        else:
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            return await self._execute_sqlite_query(query, {})

class WebAPIServerMCP(MCPServerBase):
    """MCP server for web API operations"""
    
    def __init__(self, api_configs: Dict[str, Dict[str, Any]]):
        super().__init__("webapi-server", "1.0.0")
        self.api_configs = api_configs
        self.session = None
        self._register_api_tools()
    
    def _register_api_tools(self):
        """Register web API tools"""
        # HTTP request tool
        http_tool = MCPToolDefinition(
            name="http_request",
            description="Make HTTP request to external API",
            input_schema={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                        "description": "HTTP method"
                    },
                    "url": {"type": "string", "description": "Request URL"},
                    "headers": {
                        "type": "object",
                        "description": "Request headers",
                        "default": {}
                    },
                    "data": {
                        "type": "object",
                        "description": "Request body data",
                        "default": {}
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Request timeout in seconds",
                        "default": 30
                    }
                },
                "required": ["method", "url"]
            }
        )
        self.register_tool(http_tool)
        
        # API-specific tools
        for api_name, config in self.api_configs.items():
            self._register_api_specific_tools(api_name, config)
    
    def _register_api_specific_tools(self, api_name: str, config: Dict[str, Any]):
        """Register tools for specific APIs"""
        for endpoint in config.get("endpoints", []):
            tool = MCPToolDefinition(
                name=f"{api_name}_{endpoint['name']}",
                description=endpoint.get("description", f"Call {api_name} {endpoint['name']} endpoint"),
                input_schema=endpoint.get("input_schema", {"type": "object", "properties": {}})
            )
            self.register_tool(tool)
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute web API tool"""
        import aiohttp
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        if tool_name == "http_request":
            return await self._make_http_request(arguments)
        
        # Check if it's an API-specific tool
        for api_name, config in self.api_configs.items():
            if tool_name.startswith(f"{api_name}_"):
                endpoint_name = tool_name[len(f"{api_name}_"):]
                return await self._call_api_endpoint(api_name, endpoint_name, arguments)
        
        raise Exception(f"Unknown tool: {tool_name}")
    
    async def _make_http_request(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request"""
        method = arguments["method"]
        url = arguments["url"]
        headers = arguments.get("headers", {})
        data = arguments.get("data", {})
        timeout = arguments.get("timeout", 30)
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if data else None,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                
                # Get response data
                try:
                    response_data = await response.json()
                except:
                    response_data = await response.text()
                
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "data": response_data,
                    "url": str(response.url)
                }
                
        except Exception as e:
            raise Exception(f"HTTP request failed: {str(e)}")
    
    async def _call_api_endpoint(self, api_name: str, endpoint_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call specific API endpoint"""
        config = self.api_configs[api_name]
        
        # Find endpoint configuration
        endpoint_config = None
        for endpoint in config.get("endpoints", []):
            if endpoint["name"] == endpoint_name:
                endpoint_config = endpoint
                break
        
        if not endpoint_config:
            raise Exception(f"Endpoint not found: {endpoint_name}")
        
        # Build request
        method = endpoint_config.get("method", "GET")
        url = config["base_url"] + endpoint_config["path"]
        
        # Add authentication if configured
        headers = {}
        if "auth" in config:
            auth_config = config["auth"]
            if auth_config["type"] == "bearer":
                headers["Authorization"] = f"Bearer {auth_config['token']}"
            elif auth_config["type"] == "api_key":
                headers[auth_config["header"]] = auth_config["key"]
        
        # Make request
        request_args = {
            "method": method,
            "url": url,
            "headers": headers,
            "data": arguments
        }
        
        return await self._make_http_request(request_args)
\`\`\`

### Production Server Deployment
\`\`\`python
import ssl
import logging
from pathlib import Path

class ProductionMCPServer:
    """Production-ready MCP server with security and monitoring"""
    
    def __init__(self, server_instance: MCPServerBase, config: Dict[str, Any]):
        self.server = server_instance
        self.config = config
        self.metrics = MCPServerMetrics()
        self.auth_manager = AuthenticationManager(config.get("auth", {}))
        
        # Setup logging
        self._setup_logging()
        
        # Setup SSL if configured
        self.ssl_context = self._setup_ssl()
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        log_file = log_config.get("file", "mcp_server.log")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def _setup_ssl(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context if configured"""
        ssl_config = self.config.get("ssl", {})
        
        if not ssl_config.get("enabled", False):
            return None
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            ssl_config["cert_file"],
            ssl_config["key_file"]
        )
        
        return context
    
    async def start(self):
        """Start production server"""
        host = self.config.get("host", "localhost")
        port = self.config.get("port", 8765)
        
        # Monkey patch the server's message processing for metrics and auth
        original_process_message = self.server._process_message
        
        async def enhanced_process_message(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # Record metrics
            start_time = time.time()
            method = data.get("method", "unknown")
            
            try:
                # Authentication check
                if not await self.auth_manager.authenticate(data):
                    return {
                        "jsonrpc": "2.0",
                        "id": data.get("id"),
                        "error": {
                            "code": -32600,
                            "message": "Authentication failed"
                        }
                    }
                
                # Process message
                result = await original_process_message(data)
                
                # Record success metrics
                duration = time.time() - start_time
                await self.metrics.record_request(method, "success", duration)
                
                return result
                
            except Exception as e:
                # Record error metrics
                duration = time.time() - start_time
                await self.metrics.record_request(method, "error", duration)
                raise
        
        self.server._process_message = enhanced_process_message
        
        # Start server
        await self.server.start_server(host, port)
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get server metrics"""
        return await self.metrics.get_all_metrics()

class MCPServerMetrics:
    """Metrics collection for MCP server"""
    
    def __init__(self):
        self.request_counts = {}
        self.error_counts = {}
        self.response_times = {}
        self.active_connections = 0
    
    async def record_request(self, method: str, status: str, duration: float):
        """Record request metrics"""
        # Count requests
        if method not in self.request_counts:
            self.request_counts[method] = 0
        self.request_counts[method] += 1
        
        # Count errors
        if status == "error":
            if method not in self.error_counts:
                self.error_counts[method] = 0
            self.error_counts[method] += 1
        
        # Record response times
        if method not in self.response_times:
            self.response_times[method] = []
        self.response_times[method].append(duration)
        
        # Keep only recent response times
        self.response_times[method] = self.response_times[method][-1000:]
    
    async def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        return {
            "request_counts": self.request_counts,
            "error_counts": self.error_counts,
            "error_rates": {
                method: self.error_counts.get(method, 0) / count
                for method, count in self.request_counts.items()
            },
            "average_response_times": {
                method: sum(times) / len(times) if times else 0
                for method, times in self.response_times.items()
            },
            "active_connections": self.active_connections
        }

class AuthenticationManager:
    """Authentication management for MCP server"""
    
    def __init__(self, auth_config: Dict[str, Any]):
        self.auth_config = auth_config
        self.auth_type = auth_config.get("type", "none")
        
        if self.auth_type == "api_key":
            self.valid_keys = set(auth_config.get("keys", []))
        elif self.auth_type == "jwt":
            self.jwt_secret = auth_config.get("secret")
    
    async def authenticate(self, message: Dict[str, Any]) -> bool:
        """Authenticate incoming message"""
        if self.auth_type == "none":
            return True
        
        auth_data = message.get("auth", {})
        
        if self.auth_type == "api_key":
            api_key = auth_data.get("api_key")
            return api_key in self.valid_keys
        
        elif self.auth_type == "jwt":
            import jwt
            token = auth_data.get("token")
            
            try:
                jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
                return True
            except jwt.InvalidTokenError:
                return False
        
        return False

# Example usage
async def main():
    # File system server
    fs_server = FileSystemMCPServer(allowed_paths=["/tmp", "/var/data"])
    
    # Database server
    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "user": "mcp_user",
        "password": "password",
        "database": "mcp_db"
    }
    db_server = DatabaseMCPServer(db_config)
    await db_server.initialize_database()
    
    # Production configuration
    prod_config = {
        "host": "0.0.0.0",
        "port": 8765,
        "ssl": {
            "enabled": True,
            "cert_file": "server.crt",
            "key_file": "server.key"
        },
        "auth": {
            "type": "api_key",
            "keys": ["your-api-key-here"]
        },
        "logging": {
            "level": "INFO",
            "file": "mcp_server.log"
        }
    }
    
    # Start production servers
    prod_fs_server = ProductionMCPServer(fs_server, prod_config)
    await prod_fs_server.start()

if __name__ == "__main__":
    asyncio.run(main())
\`\`\`

MCP server implementation requires careful attention to security, scalability, and reliability to provide robust tool interfaces that AI applications can depend on in production environments.`,
    date: "2024-01-22",
    tags: ["MCP", "Server Implementation", "Tool Backends", "Production Deployment"],
    readTime: "22 min read",
    category: "MCP"
  }
];