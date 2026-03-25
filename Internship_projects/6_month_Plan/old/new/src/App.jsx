import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronRight, Target, Brain, DollarSign, Dumbbell, Briefcase, Rocket, Star } from 'lucide-react';

export default function GoalTracker() {
  const [expandedMonths, setExpandedMonths] = useState({ 0: true });
  const [completedTasks, setCompletedTasks] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => { loadProgress(); }, []);

  const loadProgress = async () => {
    try {
      const saved = localStorage.getItem('goal-progress-v2');
      if (saved) setCompletedTasks(JSON.parse(saved));
    } catch (e) {}
    finally { setLoading(false); }
  };

  const saveProgress = (nc) => {
    try { localStorage.setItem('goal-progress-v2', JSON.stringify(nc)); } catch(e) {}
  };

  const toggleTask = (id) => {
    const nc = { ...completedTasks, [id]: !completedTasks[id] };
    setCompletedTasks(nc); saveProgress(nc);
  };

  const toggleMonth = (i) => setExpandedMonths(p => ({ ...p, [i]: !p[i] }));

  const roadmap = [
    {
      month: 1, title: "Get It Running — First GitHub Commit", date: "March 2026",
      badge: "🚀 Start here TODAY", badgeColor: "bg-green-100 text-green-700",
      goal: "By end of March: YOLOv8 + DeepSORT running on a real video, pushed to GitHub.",
      categories: [
        { icon: Brain, name: "Week 1–2: Detection Basics", color: "text-purple-600", tasks: [
          { id: "m1-1", text: "Install YOLOv8 via Ultralytics, run on a sample video", resource: "pip install ultralytics → yolo predict" },
          { id: "m1-2", text: "Understand YOLOv8 output: boxes, scores, class IDs", resource: "Ultralytics docs" },
          { id: "m1-3", text: "Run on your own video (street cam, phone footage, YouTube clip)", resource: "OpenCV VideoCapture" },
          { id: "m1-4", text: "Read YOLOv9 paper abstract + intro — understand PGI/GELAN conceptually", resource: "arxiv.org/abs/2402.13616" },
        ]},
        { icon: Brain, name: "Week 3–4: Add Tracking", color: "text-indigo-600", tasks: [
          { id: "m1-5", text: "Clone boxmot, integrate DeepSORT with your YOLOv8 output", resource: "github: mikel-brostrom/boxmot" },
          { id: "m1-6", text: "Read SORT paper (only 6 pages) — Kalman filter + Hungarian matching", resource: "arxiv.org/abs/1602.00763" },
          { id: "m1-7", text: "Get stable track IDs showing on video — bounding boxes with persistent IDs", resource: "OpenCV putText, rectangle" },
          { id: "m1-8", text: "Push to GitHub with README and a demo GIF", resource: "→ Your first portfolio piece" },
          { id: "m1-9", text: "Learn MOT metrics conceptually: MOTA = tracking accuracy, IDF1 = ID consistency", resource: "MOTChallenge website" },
        ]},
        { icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
          { id: "m1-fin-1", text: "Track expenses, set savings target: $500-750/month", resource: "Automate transfers" },
          { id: "m1-fin-2", text: "Save first installment toward $3,000-4,000 emergency fund", resource: "High-yield savings" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m1-h1", text: "Establish 3x/week exercise routine (30-45 min)", resource: "Any activity — just start" },
        ]},
      ]
    },
    {
      month: 2, title: "Understand One Algorithm Deeply + Build the Pipeline", date: "April 2026",
      badge: "📦 First real project", badgeColor: "bg-blue-100 text-blue-700",
      goal: "By end of April: ByteTrack on MOT17 with logged metrics. Video pipeline complete.",
      categories: [
        { icon: Brain, name: "Week 1–2: ByteTrack Deep Dive", color: "text-purple-600", tasks: [
          { id: "m2-1", text: "Read ByteTrack paper fully — it's short and practical", resource: "arxiv.org/abs/2110.06864" },
          { id: "m2-2", text: "Swap DeepSORT for ByteTrack in your pipeline using boxmot", resource: "github: mikel-brostrom/boxmot" },
          { id: "m2-3", text: "Download MOT17 dataset (or subset), run your tracker on it", resource: "MOTChallenge download page" },
          { id: "m2-4", text: "Log MOTA and IDF1 scores — understand what makes them go up/down", resource: "py-motmetrics library" },
        ]},
        { icon: Brain, name: "Week 3–4: Video Pipeline", color: "text-indigo-600", tasks: [
          { id: "m2-5", text: "Build clean pipeline: read → detect → track → annotate → write", resource: "OpenCV VideoWriter, FFmpeg" },
          { id: "m2-6", text: "Handle edge cases: empty frames, no detections, video end", resource: "Basic error handling" },
          { id: "m2-7", text: "Choose your use case: traffic OR retail/people counting (pick ONE)", resource: "Traffic = easier data, retail = more practical" },
          { id: "m2-8", text: "Draw trajectory lines behind tracked objects", resource: "cv2.polylines with deque history" },
          { id: "m2-9", text: "Update GitHub: clean structure, MOT17 benchmark results in README", resource: "→ Recruiters look at READMEs" },
        ]},
        { icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
          { id: "m2-fin-1", text: "Save second installment (total: $1,000-1,500)", resource: "Stay consistent" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m2-h1", text: "Continue 3x/week routine, add variety", resource: "Mix cardio + strength" },
        ]},
      ]
    },
    {
      month: 3, title: "Make It Useful — Analytics + Simple API", date: "May 2026",
      badge: "📊 Something to demo", badgeColor: "bg-purple-100 text-purple-700",
      goal: "By end of May: Working demo with analytics (counts, heatmap) + /upload and /results API.",
      categories: [
        { icon: Brain, name: "Week 1–2: Domain Analytics", color: "text-purple-600", tasks: [
          { id: "m3-1", text: "Add object counter: total counts per class per video", resource: "Simple dict counter" },
          { id: "m3-2", text: "Add zone-crossing detection: draw a line, count objects that cross it", resource: "Line intersection math" },
          { id: "m3-3", text: "Generate trajectory heatmap as output image", resource: "numpy + matplotlib imshow" },
          { id: "m3-4", text: "Export results as JSON: {track_id, class, frames_seen, trajectory}", resource: "json.dump" },
        ]},
        { icon: Briefcase, name: "Week 3–4: Basic API", color: "text-blue-600", tasks: [
          { id: "m3-5", text: "Build FastAPI with POST /upload and GET /results/{job_id}", resource: "fastapi.tiangolo.com" },
          { id: "m3-6", text: "Process video in background (simple threading, no Celery yet)", resource: "BackgroundTasks in FastAPI" },
          { id: "m3-7", text: "Profile pipeline with cProfile — find #1 bottleneck and fix it", resource: "python -m cProfile" },
          { id: "m3-8", text: "Record 2-3 min demo video of full pipeline in action", resource: "OBS or Loom — for portfolio" },
        ]},
        { icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
          { id: "m3-fin-1", text: "Save third installment (total: $1,500-2,250)", resource: "Halfway there!" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m3-h1", text: "Maintain 3-4x/week routine", resource: "Build the habit" },
        ]},
      ]
    },
    {
      month: 4, title: "Dockerize + Deploy — Get a Live URL", date: "June 2026",
      badge: "🌐 Live demo = internship magnet", badgeColor: "bg-orange-100 text-orange-700",
      goal: "By end of June: App is live on the internet with a shareable URL. Start applying.",
      categories: [
        { icon: Briefcase, name: "Week 1–2: Containerize", color: "text-blue-600", tasks: [
          { id: "m4-1", text: "Write a Dockerfile for your FastAPI + tracker app", resource: "Multi-stage build to keep image small" },
          { id: "m4-2", text: "Set up docker-compose for local dev", resource: "docker-compose.yml" },
          { id: "m4-3", text: "Add structured logging and basic error handling", resource: "Python logging module" },
          { id: "m4-4", text: "Auto-generate API docs with FastAPI's built-in Swagger UI", resource: "/docs endpoint — free and impressive" },
        ]},
        { icon: Briefcase, name: "Week 3–4: Deploy & Apply", color: "text-indigo-600", tasks: [
          { id: "m4-5", text: "Deploy to Railway, Render, or EC2 free tier — get a live URL", resource: "Railway is easiest; EC2 gives more CV cred" },
          { id: "m4-6", text: "Add GitHub Actions: auto-deploy on push to main", resource: "CI/CD basics" },
          { id: "m4-7", text: "Build a simple Streamlit dashboard showing analytics (counts, heatmap)", resource: "streamlit.io" },
          { id: "m4-8", text: "🎯 START APPLYING: AI/CV/ML internships for Autumn. Apply to 10+ companies this month", resource: "LinkedIn, Glassdoor, company career pages" },
          { id: "m4-9", text: "Polish LinkedIn: add project with live URL, demo GIF, benchmark numbers", resource: "Recruiters will Google your name" },
        ]},
        { icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
          { id: "m4-fin-1", text: "Save fourth installment (total: $2,000-3,000)", resource: "Getting close!" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m4-h1", text: "Exercise 3-4x/week — should feel natural now", resource: "Consistency matters" },
        ]},
      ]
    },
    {
      month: 5, title: "Polish, Interview Prep + Advanced Features", date: "July 2026",
      badge: "🎯 Interview season", badgeColor: "bg-red-100 text-red-700",
      goal: "By end of July: Interview-ready. Strong GitHub, blog post written, LC + CV interview prep done.",
      categories: [
        { icon: Brain, name: "Week 1–2: Deepen Technical Knowledge", color: "text-purple-600", tasks: [
          { id: "m5-1", text: "Read DeepSORT paper fully — appearance features + Re-ID", resource: "arxiv.org/abs/1703.07402" },
          { id: "m5-2", text: "Add TensorRT or ONNX export to your pipeline (speed boost)", resource: "Ultralytics export guide" },
          { id: "m5-3", text: "Try BoT-SORT or StrongSORT — compare metrics vs ByteTrack", resource: "boxmot tracker swap" },
          { id: "m5-4", text: "Add MLflow experiment tracking: log FPS, MOTA, IDF1 per run", resource: "mlflow.org" },
        ]},
        { icon: Briefcase, name: "Week 3–4: Interview Prep", color: "text-blue-600", tasks: [
          { id: "m5-5", text: "Write technical blog post: 'How I built a real-time MOT system' with benchmarks", resource: "Medium or Dev.to — shows communication skills" },
          { id: "m5-6", text: "Practice explaining your project in 2 minutes (mock interview with a friend)", resource: "Record yourself on video" },
          { id: "m5-7", text: "Prep CV-specific questions: detection, tracking, Kalman filter, Hungarian algorithm", resource: "Papers With Code - MOT section" },
          { id: "m5-8", text: "Do 20+ LeetCode mediums (arrays, graphs, DP) — AI internships still test LC", resource: "Focus on patterns, not memorization" },
          { id: "m5-9", text: "Apply to 20+ more companies. Prioritize startups — they move faster", resource: "YC companies, LinkedIn Easy Apply" },
        ]},
        { icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
          { id: "m5-fin-1", text: "Save fifth installment (total: $2,500-3,750)", resource: "Almost there!" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m5-h1", text: "Maintain routine — it's a lifestyle now", resource: "Keep it up" },
        ]},
      ]
    },
    {
      month: 6, title: "Launch, Contribute & Land the Internship", date: "August 2026",
      badge: "🏆 Finish strong", badgeColor: "bg-yellow-100 text-yellow-700",
      goal: "By end of August: Internship offer in hand. Open source contribution. System fully documented.",
      categories: [
        { icon: Rocket, name: "Week 1–2: Open Source + Advanced Features", color: "text-purple-600", tasks: [
          { id: "m6-1", text: "Submit one PR to Ultralytics or roboflow/supervision (even docs/bug fix counts)", resource: "Shows collaboration — gold for recruiters" },
          { id: "m6-2", text: "Add one advanced feature: pose estimation OR multi-camera ReID (pick one)", resource: "YOLOv8-pose or boxmot cross-camera" },
          { id: "m6-3", text: "Add Prometheus metrics endpoint + simple Grafana dashboard for latency/FPS", resource: "Shows MLOps awareness" },
        ]},
        { icon: Briefcase, name: "Week 3–4: Land It", color: "text-blue-600", tasks: [
          { id: "m6-4", text: "Final GitHub polish: clean README, architecture diagram, demo video linked, badges", resource: "shields.io for badges" },
          { id: "m6-5", text: "Create case study: FPS benchmarks, MOTA scores, cost to run, use case impact", resource: "One-page PDF or Notion page" },
          { id: "m6-6", text: "Share project on LinkedIn + r/computervision + r/MachineLearning", resource: "Real visibility — people get hired from this" },
          { id: "m6-7", text: "Follow up on all pending applications. Send cold emails to CV/ML engineers at target companies", resource: "5-line email: who you are, what you built, link, ask for 15 min" },
          { id: "m6-8", text: "Read one new MOT/CV paper per week from ArXiv", resource: "Stay sharp for final interviews" },
        ]},
        { icon: Star, name: "Continuous Learning", color: "text-yellow-600", tasks: [
          { id: "m6-l1", text: "Contribute to open source: Ultralytics, supervision library", resource: "arxiv.org cs.CV section" },
        ]},
        { icon: DollarSign, name: "Financial — Goal Achieved! 🎉", color: "text-green-600", tasks: [
          { id: "m6-fin-1", text: "Final savings push (total: $3,000-4,000+ emergency fund!)", resource: "Mission accomplished!" },
          { id: "m6-fin-2", text: "Plan next financial milestone: 6-month fund or investments", resource: "Continue growing" },
        ]},
        { icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
          { id: "m6-h1", text: "Celebrate health gains — you've built a real habit", resource: "Set the next fitness challenge" },
        ]},
      ]
    },
  ];

  const getMonthProgress = (month) => {
    const all = month.categories.flatMap(c => c.tasks);
    return Math.round((all.filter(t => completedTasks[t.id]).length / all.length) * 100);
  };

  const getTotalProgress = () => {
    const all = roadmap.flatMap(m => m.categories.flatMap(c => c.tasks));
    return Math.round((all.filter(t => completedTasks[t.id]).length / all.length) * 100);
  };

  if (loading) return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
      <div className="text-slate-600">Loading your progress...</div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 md:p-8">
      <div className="max-w-5xl mx-auto">
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Target className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl md:text-4xl font-bold text-slate-900">MOT Specialization — Realistic Edition</h1>
          </div>
          <p className="text-slate-500 mb-1 text-sm">March – August 2026 · Student pace · 1-2 hrs/day · Autumn internship target</p>
          <p className="text-indigo-600 font-medium text-sm mb-4">Each month ends with ONE thing you can show a recruiter.</p>
          <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">Overall Progress</span>
              <span className="text-2xl font-bold text-indigo-600">{getTotalProgress()}%</span>
            </div>
            <div className="w-full bg-slate-200 rounded-full h-3">
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 h-3 rounded-full transition-all duration-500" style={{ width: `${getTotalProgress()}%` }} />
            </div>
          </div>
        </div>

        <div className="space-y-4">
          {roadmap.map((month, index) => {
            const isExpanded = expandedMonths[index];
            const progress = getMonthProgress(month);
            return (
              <div key={index} className="bg-white rounded-2xl shadow-lg overflow-hidden">
                <button onClick={() => toggleMonth(index)} className="w-full p-6 flex items-center justify-between hover:bg-slate-50 transition-colors">
                  <div className="flex items-center gap-4">
                    <div className="bg-indigo-100 rounded-full w-12 h-12 flex items-center justify-center flex-shrink-0">
                      <span className="text-indigo-700 font-bold text-lg">M{month.month}</span>
                    </div>
                    <div className="text-left">
                      <div className="flex items-center gap-2 flex-wrap">
                        <h2 className="text-xl font-bold text-slate-900">{month.title}</h2>
                        <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${month.badgeColor}`}>{month.badge}</span>
                      </div>
                      <p className="text-sm text-slate-500">{month.date}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4 flex-shrink-0">
                    <div className="text-right hidden sm:block">
                      <div className="text-sm font-medium text-slate-600">{progress}% complete</div>
                      <div className="w-32 bg-slate-200 rounded-full h-2 mt-1">
                        <div className="bg-indigo-600 h-2 rounded-full transition-all duration-300" style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                    {isExpanded ? <ChevronDown className="w-6 h-6 text-slate-400" /> : <ChevronRight className="w-6 h-6 text-slate-400" />}
                  </div>
                </button>

                {isExpanded && (
                  <div className="px-6 pb-6 space-y-5">
                    <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-3">
                      <p className="text-sm font-semibold text-indigo-700">🎯 Month goal: {month.goal}</p>
                    </div>
                    {month.categories.map((cat, ci) => {
                      const Icon = cat.icon;
                      return (
                        <div key={ci} className="border-l-4 border-slate-200 pl-4">
                          <div className="flex items-center gap-2 mb-3">
                            <Icon className={`w-5 h-5 ${cat.color}`} />
                            <h3 className={`font-semibold ${cat.color}`}>{cat.name}</h3>
                          </div>
                          <div className="space-y-2">
                            {cat.tasks.map((task) => (
                              <div key={task.id} className="flex items-start gap-3 p-3 rounded-lg hover:bg-slate-50 transition-colors group">
                                <button onClick={() => toggleTask(task.id)} className="mt-0.5 flex-shrink-0">
                                  {completedTasks[task.id]
                                    ? <CheckCircle2 className="w-5 h-5 text-green-600" />
                                    : <Circle className="w-5 h-5 text-slate-300 group-hover:text-slate-400" />}
                                </button>
                                <div className="flex-1">
                                  <p className={`text-sm ${completedTasks[task.id] ? 'line-through text-slate-400' : 'text-slate-700'}`}>{task.text}</p>
                                  {task.resource && <p className="text-xs text-slate-500 mt-0.5">💡 {task.resource}</p>}
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

        <div className="bg-white rounded-2xl shadow-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-slate-900 mb-4">🔗 Essential Resources</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">Core Tracking</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• boxmot (mikel-brostrom) — plug-and-play trackers</li>
                <li>• ByteTrack paper (2021) — start here</li>
                <li>• SORT paper (2016) — only 6 pages</li>
                <li>• MOTChallenge benchmark + metrics</li>
                <li>• py-motmetrics — evaluate your tracker</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">Production Stack</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• FastAPI — backend API</li>
                <li>• Docker + docker-compose</li>
                <li>• Railway or Render — free deployment</li>
                <li>• Streamlit — quick dashboard</li>
                <li>• MLflow — experiment tracking</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-indigo-600 mb-2">Detection</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• Ultralytics YOLOv8/v9/v10</li>
                <li>• roboflow/supervision — annotation utils</li>
                <li>• ONNX export for optimization</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-green-600 mb-2">Job Hunting</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• LinkedIn Easy Apply (AI/CV/ML intern)</li>
                <li>• YC company list — ycombinator.com/companies</li>
                <li>• r/cscareerquestions, r/computervision</li>
                <li>• Papers With Code — know current SOTA</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="text-center text-slate-500 text-sm mt-8 pb-4">
          <p>Progress saves automatically. Keep M1 open and start ticking boxes <strong>today</strong>.</p>
          <p className="mt-1">🏁 One working demo beats ten half-read papers.</p>
        </div>
      </div>
    </div>
  );
}
