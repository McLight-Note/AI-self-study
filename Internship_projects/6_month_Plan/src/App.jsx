import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronRight, Target, Brain, DollarSign, Dumbbell, Briefcase } from 'lucide-react';

export default function GoalTracker() {
  const [expandedMonths, setExpandedMonths] = useState({ 0: true });
  const [completedTasks, setCompletedTasks] = useState({});
  const [loading, setLoading] = useState(true);

  // Load completed tasks from storage
  useEffect(() => {
    loadProgress();
  }, []);

  const loadProgress = async () => {
    try {
      const saved = localStorage.getItem('goal-progress');
      if (saved) {
        setCompletedTasks(JSON.parse(saved));
      }
    } catch (error) {
      console.log('No saved progress found, starting fresh');
    } finally {
      setLoading(false);
    }
  };

  const saveProgress = async (newCompleted) => {
    try {
      localStorage.setItem('goal-progress', JSON.stringify(newCompleted));
    } catch (error) {
      console.error('Failed to save progress:', error);
    }
  };

  const toggleTask = (taskId) => {
    const newCompleted = {
      ...completedTasks,
      [taskId]: !completedTasks[taskId]
    };
    setCompletedTasks(newCompleted);
    saveProgress(newCompleted);
  };

  const toggleMonth = (index) => {
    setExpandedMonths(prev => ({
      ...prev,
      [index]: !prev[index]
    }));
  };

  const roadmap = [
    {
      month: 1,
      title: "Advanced Detection & MOT Foundations",
      date: "March 2026",
      categories: [
        {
          icon: Brain,
          name: "Modern Object Detection",
          color: "text-purple-600",
          tasks: [
            { id: "m1-ai-1", text: "Study YOLOv9/v10 architecture improvements (PGI, GELAN)", resource: "YOLOv9 paper, GitHub" },
            { id: "m1-ai-2", text: "Implement RT-DETR (real-time transformer detector)", resource: "Ultralytics RT-DETR" },
            { id: "m1-ai-3", text: "Compare YOLO vs transformer-based detectors on video", resource: "Benchmark FPS, accuracy" },
            { id: "m1-ai-4", text: "Optimize inference pipeline: TensorRT, ONNX, half-precision", resource: "TensorRT docs" },
            { id: "m1-ai-5", text: "Study ReID (re-identification) fundamentals for tracking", resource: "Deep ReID papers" }
          ]
        },
        {
          icon: Brain,
          name: "Multi-Object Tracking Theory",
          color: "text-indigo-600",
          tasks: [
            { id: "m1-mot-1", text: "Deep dive: SORT algorithm (Kalman filter + Hungarian matching)", resource: "SORT paper 2016" },
            { id: "m1-mot-2", text: "Study DeepSORT architecture (appearance features + motion)", resource: "DeepSORT paper" },
            { id: "m1-mot-3", text: "Learn MOT metrics: MOTA, MOTP, IDF1, HOTA", resource: "MOTChallenge metrics" },
            { id: "m1-mot-4", text: "Implement basic SORT tracker from scratch (educational)", resource: "NumPy + SciPy" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial",
          color: "text-green-600",
          tasks: [
            { id: "m1-fin-1", text: "Track expenses, set savings target: $500-750/month", resource: "Automate transfers" },
            { id: "m1-fin-2", text: "Save first installment toward $3,000-4,000 emergency fund", resource: "High-yield savings" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m1-health-1", text: "Establish 3x/week exercise routine (30-45 min)", resource: "Any activity" }
          ]
        }
      ]
    },
    {
      month: 2,
      title: "Advanced MOT Algorithms & Video Processing",
      date: "April 2026",
      categories: [
        {
          icon: Brain,
          name: "State-of-the-Art MOT",
          color: "text-purple-600",
          tasks: [
            { id: "m2-ai-1", text: "Implement ByteTrack (low-confidence detection association)", resource: "ByteTrack paper, GitHub" },
            { id: "m2-ai-2", text: "Study BoT-SORT (appearance + motion + camera motion)", resource: "BoT-SORT paper 2022" },
            { id: "m2-ai-3", text: "Implement StrongSORT or OC-SORT (latest algorithms)", resource: "GitHub implementations" },
            { id: "m2-ai-4", text: "Compare tracking algorithms on MOT17/MOT20 benchmark", resource: "MOTChallenge dataset" },
            { id: "m2-ai-5", text: "Handle occlusions, ID switches, crowded scenes", resource: "Advanced techniques" }
          ]
        },
        {
          icon: Brain,
          name: "Video Analysis Pipeline",
          color: "text-indigo-600",
          tasks: [
            { id: "m2-vid-1", text: "Build efficient video processing pipeline (frame extraction, batching)", resource: "OpenCV, FFmpeg" },
            { id: "m2-vid-2", text: "Implement temporal smoothing and post-processing", resource: "Track interpolation" },
            { id: "m2-vid-3", text: "Add trajectory prediction and motion forecasting", resource: "Kalman filter variants" }
          ]
        },
        {
          icon: Briefcase,
          name: "Production System Design",
          color: "text-blue-600",
          tasks: [
            { id: "m2-sys-1", text: "Design system architecture: video ingestion → detection → tracking → output", resource: "Microservices pattern" },
            { id: "m2-sys-2", text: "Choose use case: surveillance, sports, traffic, retail analytics, etc.", resource: "Define requirements" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial",
          color: "text-green-600",
          tasks: [
            { id: "m2-fin-1", text: "Save second installment (total: $1,000-1,500)", resource: "Stay consistent" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m2-health-1", text: "Continue 3x/week routine, add variety", resource: "Mix cardio + strength" }
          ]
        }
      ]
    },
    {
      month: 3,
      title: "Real-time Optimization & API Development",
      date: "May 2026",
      categories: [
        {
          icon: Brain,
          name: "Performance Optimization",
          color: "text-purple-600",
          tasks: [
            { id: "m3-perf-1", text: "Profile entire pipeline: identify bottlenecks (CPU/GPU)", resource: "cProfile, py-spy, nvprof" },
            { id: "m3-perf-2", text: "Implement frame skipping and adaptive FPS for real-time", resource: "Dynamic processing" },
            { id: "m3-perf-3", text: "Optimize data movement: minimize CPU-GPU transfers", resource: "CUDA streams" },
            { id: "m3-perf-4", text: "Add multi-threading for video I/O (separate from inference)", resource: "queue.Queue, threading" },
            { id: "m3-perf-5", text: "Achieve 30+ FPS on 1080p video (or 15+ FPS on 4K)", resource: "Real-time target" }
          ]
        },
        {
          icon: Briefcase,
          name: "API & Backend Development",
          color: "text-blue-600",
          tasks: [
            { id: "m3-api-1", text: "Build FastAPI REST endpoints: /upload, /process, /results, /stream", resource: "FastAPI docs" },
            { id: "m3-api-2", text: "Implement video upload handling and storage", resource: "S3, local storage" },
            { id: "m3-api-3", text: "Add WebSocket for real-time streaming results", resource: "WebSocket API" },
            { id: "m3-api-4", text: "Implement job queue for async processing (Celery/Redis)", resource: "Background tasks" },
            { id: "m3-api-5", text: "Add authentication and rate limiting", resource: "JWT, API keys" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial",
          color: "text-green-600",
          tasks: [
            { id: "m3-fin-1", text: "Save third installment (total: $1,500-2,250)", resource: "Halfway there!" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m3-health-1", text: "Maintain 3-4x/week routine", resource: "Build habit" }
          ]
        }
      ]
    },
    {
      month: 4,
      title: "Analytics, Visualization & Containerization",
      date: "June 2026",
      categories: [
        {
          icon: Brain,
          name: "Advanced Analytics",
          color: "text-purple-600",
          tasks: [
            { id: "m4-ana-1", text: "Build analytics layer: count objects, dwell time, trajectory heatmaps", resource: "Custom metrics" },
            { id: "m4-ana-2", text: "Implement zone-based analytics (entry/exit, restricted areas)", resource: "Polygon containment" },
            { id: "m4-ana-3", text: "Add activity recognition or anomaly detection (optional advanced)", resource: "Behavior analysis" },
            { id: "m4-ana-4", text: "Create dashboard with real-time statistics", resource: "Plotly Dash, Streamlit" },
            { id: "m4-ana-5", text: "Export results: JSON, CSV, annotated video", resource: "Multiple formats" }
          ]
        },
        {
          icon: Briefcase,
          name: "Production Infrastructure",
          color: "text-blue-600",
          tasks: [
            { id: "m4-infra-1", text: "Dockerize entire application (detector + tracker + API)", resource: "Multi-stage Dockerfile" },
            { id: "m4-infra-2", text: "Set up docker-compose for local development", resource: "docker-compose.yml" },
            { id: "m4-infra-3", text: "Add health checks and logging (structured logs)", resource: "Python logging, ELK stack" },
            { id: "m4-infra-4", text: "Write comprehensive API documentation (OpenAPI/Swagger)", resource: "Auto-generated docs" },
            { id: "m4-infra-5", text: "Implement error handling and graceful degradation", resource: "Robust system" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial",
          color: "text-green-600",
          tasks: [
            { id: "m4-fin-1", text: "Save fourth installment (total: $2,000-3,000)", resource: "Getting close!" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m4-health-1", text: "Exercise 3-4x/week - should feel natural", resource: "Consistency matters" }
          ]
        }
      ]
    },
    {
      month: 5,
      title: "Cloud Deployment & Scalability",
      date: "July 2026",
      categories: [
        {
          icon: Brain,
          name: "MLOps & Monitoring",
          color: "text-purple-600",
          tasks: [
            { id: "m5-ops-1", text: "Implement model versioning and A/B testing framework", resource: "MLflow, DVC" },
            { id: "m5-ops-2", text: "Add inference monitoring: latency, throughput, accuracy", resource: "Prometheus, Grafana" },
            { id: "m5-ops-3", text: "Set up data drift detection for production", resource: "Evidently AI" },
            { id: "m5-ops-4", text: "Create model retraining pipeline (batch or incremental)", resource: "Automated retraining" }
          ]
        },
        {
          icon: Briefcase,
          name: "Cloud Deployment",
          color: "text-blue-600",
          tasks: [
            { id: "m5-cloud-1", text: "Deploy to cloud: AWS (EC2 + Lambda) or GCP (Compute + Functions)", resource: "Choose provider" },
            { id: "m5-cloud-2", text: "Set up auto-scaling based on load", resource: "Kubernetes or cloud auto-scaling" },
            { id: "m5-cloud-3", text: "Implement CDN for video delivery and caching", resource: "CloudFront, CloudFlare" },
            { id: "m5-cloud-4", text: "Add CI/CD pipeline: GitHub Actions → Docker → Cloud", resource: "Automated deployment" },
            { id: "m5-cloud-5", text: "Conduct load testing: handle 10+ concurrent video streams", resource: "Locust, k6" }
          ]
        },
        {
          icon: Briefcase,
          name: "Portfolio & Documentation",
          color: "text-blue-600",
          tasks: [
            { id: "m5-port-1", text: "Create compelling demo: show before/after, metrics, use cases", resource: "Video demo" },
            { id: "m5-port-2", text: "Write technical documentation: architecture, setup, API usage", resource: "Comprehensive docs" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial",
          color: "text-green-600",
          tasks: [
            { id: "m5-fin-1", text: "Save fifth installment (total: $2,500-3,750)", resource: "Almost there!" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m5-health-1", text: "Maintain routine - lifestyle now", resource: "Keep it up" }
          ]
        }
      ]
    },
    {
      month: 6,
      title: "Production Launch & Advanced Features",
      date: "August 2026",
      categories: [
        {
          icon: Brain,
          name: "Advanced Features (Optional)",
          color: "text-purple-600",
          tasks: [
            { id: "m6-adv-1", text: "Add multi-camera support and cross-camera tracking", resource: "ReID across cameras" },
            { id: "m6-adv-2", text: "Implement pose estimation integration (track + pose)", resource: "YOLOv8-pose" },
            { id: "m6-adv-3", text: "Add object attribute classification (color, type, etc.)", resource: "Multi-task learning" },
            { id: "m6-adv-4", text: "Explore 3D tracking or depth estimation", resource: "Advanced research" }
          ]
        },
        {
          icon: Briefcase,
          name: "Launch & Marketing",
          color: "text-blue-600",
          tasks: [
            { id: "m6-launch-1", text: "Polish GitHub repo: clean code, comprehensive README, badges", resource: "Professional presentation" },
            { id: "m6-launch-2", text: "Write technical blog post: architecture, challenges, solutions", resource: "Medium, Dev.to" },
            { id: "m6-launch-3", text: "Create case study with metrics: FPS, accuracy, cost analysis", resource: "Show ROI" },
            { id: "m6-launch-4", text: "Share on LinkedIn, Twitter, Reddit (r/computervision, r/MachineLearning)", resource: "Build visibility" },
            { id: "m6-launch-5", text: "Apply to production ML/CV engineer roles with portfolio", resource: "Leverage your work" },
            { id: "m6-launch-6", text: "Consider open-sourcing or creating SaaS offering", resource: "Monetization options" }
          ]
        },
        {
          icon: Brain,
          name: "Continuous Learning",
          color: "text-purple-600",
          tasks: [
            { id: "m6-learn-1", text: "Read latest MOT papers (2025-2026) from ArXiv", resource: "Stay current" },
            { id: "m6-learn-2", text: "Contribute to open source: Ultralytics, supervision library", resource: "Give back" }
          ]
        },
        {
          icon: DollarSign,
          name: "Financial - Goal Achieved!",
          color: "text-green-600",
          tasks: [
            { id: "m6-fin-1", text: "Final savings push (total: $3,000-4,000+ emergency fund!)", resource: "Mission accomplished!" },
            { id: "m6-fin-2", text: "Plan next financial milestone: 6-month fund or investments", resource: "Continue growing" }
          ]
        },
        {
          icon: Dumbbell,
          name: "Health",
          color: "text-red-600",
          tasks: [
            { id: "m6-health-1", text: "Celebrate health gains, set next fitness challenge", resource: "You did it!" }
          ]
        }
      ]
    }
  ];

  const getMonthProgress = (month) => {
    const allTasks = month.categories.flatMap(cat => cat.tasks);
    const completed = allTasks.filter(task => completedTasks[task.id]).length;
    return Math.round((completed / allTasks.length) * 100);
  };

  const getTotalProgress = () => {
    const allTasks = roadmap.flatMap(month => 
      month.categories.flatMap(cat => cat.tasks)
    );
    const completed = allTasks.filter(task => completedTasks[task.id]).length;
    return Math.round((completed / allTasks.length) * 100);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-slate-600">Loading your progress...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8 mb-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl md:text-4xl font-bold text-slate-900">
              Advanced MOT Specialization Plan
            </h1>
          </div>
          <p className="text-slate-600 mb-4">
            March - August 2026 | Master Multi-Object Tracking, Build Production-Ready Video Analysis System
          </p>
          
          {/* Overall Progress */}
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">Overall Progress</span>
              <span className="text-2xl font-bold text-indigo-600">{getTotalProgress()}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-3">
              <div 
                className="bg-gradient-to-r from-indigo-600 to-purple-600 h-3 rounded-full transition-all duration-500"
                style={{ width: `${getTotalProgress()}%` }}
              />
            </div>
          </div>
        </div>

        {/* Monthly Roadmap */}
        <div className="space-y-4">
          {roadmap.map((month, index) => {
            const isExpanded = expandedMonths[index];
            const progress = getMonthProgress(month);
            
            return (
              <div key={index} className="bg-white rounded-2xl shadow-lg overflow-hidden">
                {/* Month Header */}
                <button
                  onClick={() => toggleMonth(index)}
                  className="w-full p-6 flex items-center justify-between hover:bg-slate-50 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="bg-indigo-100 rounded-full w-12 h-12 flex items-center justify-center">
                      <span className="text-indigo-700 font-bold text-lg">M{month.month}</span>
                    </div>
                    <div className="text-left">
                      <h2 className="text-xl font-bold text-slate-900">{month.title}</h2>
                      <p className="text-sm text-slate-500">{month.date}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className="text-sm font-medium text-slate-600">{progress}% complete</div>
                      <div className="w-32 bg-slate-200 rounded-full h-2 mt-1">
                        <div 
                          className="bg-indigo-600 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${progress}%` }}
                        />
                      </div>
                    </div>
                    {isExpanded ? (
                      <ChevronDown className="w-6 h-6 text-slate-400" />
                    ) : (
                      <ChevronRight className="w-6 h-6 text-slate-400" />
                    )}
                  </div>
                </button>

                {/* Month Content */}
                {isExpanded && (
                  <div className="p-6 pt-0 space-y-6">
                    {month.categories.map((category, catIndex) => {
                      const Icon = category.icon;
                      return (
                        <div key={catIndex} className="border-l-4 border-slate-200 pl-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Icon className={`w-5 h-5 ${category.color}`} />
                            <h3 className={`font-semibold ${category.color}`}>
                              {category.name}
                            </h3>
                          </div>
                          <div className="space-y-2">
                            {category.tasks.map((task) => (
                              <div
                                key={task.id}
                                className="flex items-start gap-3 p-3 rounded-lg hover:bg-slate-50 transition-colors group"
                              >
                                <button
                                  onClick={() => toggleTask(task.id)}
                                  className="mt-0.5 flex-shrink-0"
                                >
                                  {completedTasks[task.id] ? (
                                    <CheckCircle2 className="w-5 h-5 text-green-600" />
                                  ) : (
                                    <Circle className="w-5 h-5 text-slate-300 group-hover:text-slate-400" />
                                  )}
                                </button>
                                <div className="flex-1">
                                  <p className={`text-sm ${completedTasks[task.id] ? 'line-through text-slate-400' : 'text-slate-700'}`}>
                                    {task.text}
                                  </p>
                                  {task.resource && (
                                    <p className="text-xs text-slate-500 mt-1">
                                      💡 {task.resource}
                                    </p>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Key Resources */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-slate-900 mb-4">🔗 Essential Resources</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">Multi-Object Tracking</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• ByteTrack (GitHub: ifzhang/ByteTrack)</li>
                <li>• BoT-SORT paper & implementation</li>
                <li>• MOTChallenge benchmark & metrics</li>
                <li>• Supervision library (roboflow/supervision)</li>
                <li>• Papers: SORT, DeepSORT, StrongSORT</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">Production & Deployment</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• FastAPI for REST APIs</li>
                <li>• TensorRT for optimization</li>
                <li>• Docker & docker-compose</li>
                <li>• Celery + Redis for job queues</li>
                <li>• MLflow for experiment tracking</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-indigo-600 mb-2">Detection Models</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• YOLOv9/v10 (Ultralytics)</li>
                <li>• RT-DETR (real-time transformer)</li>
                <li>• ONNX & TensorRT optimization</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-green-600 mb-2">Community & Learning</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• r/computervision, r/MachineLearning</li>
                <li>• Computer Vision Discord servers</li>
                <li>• Papers with Code - MOT section</li>
                <li>• ArXiv for latest research</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center text-slate-500 text-sm mt-8">
          <p>Your progress is saved automatically. Check off tasks as you complete them!</p>
          <p className="mt-2">🚀 From algorithms to production - you're building real-world AI systems.</p>
        </div>
      </div>
    </div>
  );
}
