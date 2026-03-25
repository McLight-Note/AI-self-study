import React, { useState, useEffect } from 'react';
import { CheckCircle2, Circle, ChevronDown, ChevronRight, Target, Brain, DollarSign, Dumbbell, Briefcase, Rocket, Star, BookOpen, Calendar, Code2 } from 'lucide-react';

export default function GoalTracker() {
  const [expandedMonths, setExpandedMonths] = useState({ 0: true });
  const [expandedSchedule, setExpandedSchedule] = useState(false);
  const [completedTasks, setCompletedTasks] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => { loadProgress(); }, []);

  const loadProgress = async () => {
    try {
      const saved = localStorage.getItem('goal-progress-v3');
      if (saved) setCompletedTasks(JSON.parse(saved));
    } catch (e) {}
    finally { setLoading(false); }
  };

  const saveProgress = (nc) => {
    try { localStorage.setItem('goal-progress-v3', JSON.stringify(nc)); } catch(e) {}
  };

  const toggleTask = (id) => {
    const nc = { ...completedTasks, [id]: !completedTasks[id] };
    setCompletedTasks(nc); saveProgress(nc);
  };

  const toggleMonth = (i) => setExpandedMonths(p => ({ ...p, [i]: !p[i] }));

  const weeklySchedule = [
    { day: "화 / 수 / 목", label: "Uni days", color: "bg-blue-50 border-blue-200", items: [
      "1–1.5hr AI project after last class",
      "20 min Korean — new vocab (Anki) + grammar point from current chapter",
      "University coursework first if exams are close",
    ]},
    { day: "월 / 금요일", label: "LeetCode + Power day", color: "bg-green-50 border-green-200", items: [
      "Mon: 2hr LeetCode — 1–2 problems, focus on understanding the pattern",
      "Fri: 3–4hr AI project deep work (your main session)",
      "Fri: 1 full Korean chapter — read, vocab list, exercises",
      "Weekly review: what did you actually ship this week?",
    ]},
    { day: "토 / 일 / 월", label: "12hr work shifts", color: "bg-orange-50 border-orange-200", items: [
      "Commute / break: Anki flashcard reviews only (10–15 min)",
      "Before sleep: 20 min — listen to 서울대 audio or review grammar notes",
      "No AI project. Rest is also training.",
    ]},
  ];

  // 서울대 한국어 3A chapters
  const korean3A = [
    "1과 신입생 환영회를 한다고 해요 — indirect speech (다고 해요)",
    "2과 방을 바꿔 달라고 해 봐 — requests + indirect commands",
    "3과 비가 이렇게 많이 올 줄 몰랐어요 — unexpected outcomes (ㄹ 줄 몰랐다)",
    "4과 먹어 보니까 맛있던데요 — past experience contrast (던데요)",
    "5과 입어 보고 살걸 그랬어요 — regret (ㄹ걸 그랬다)",
    "6과 일요일에는 아무 약속도 없어요 — negative + 아무",
    "7과 껐다가 다시 켜 보세요 — sequential action (았다가)",
    "8과 교통사고가 났다고요? — surprised repetition + reported speech",
    "9과 한글날에 대해 들어 봤어요? — topic introduction (에 대해)",
  ];

  const korean3B = [
    "1과 — new grammar patterns, vocab set 1",
    "2과 — new grammar patterns, vocab set 2",
    "3과 — new grammar patterns, vocab set 3",
    "4과 — new grammar patterns, vocab set 4",
    "5과 — new grammar patterns, vocab set 5",
    "6과 — new grammar patterns, vocab set 6",
    "7과 — new grammar patterns, vocab set 7",
    "8과 — new grammar patterns, vocab set 8",
    "9과 — new grammar patterns, vocab set 9",
  ];

  const roadmap = [
    {
      month: 1, title: "Get It Running — First GitHub Commit", date: "March 2026",
      badge: "🚀 Start TODAY", badgeColor: "bg-green-100 text-green-700",
      goal: "AI: YOLOv8 + DeepSORT on a real video, pushed to GitHub. Korean: Complete 3A chapters 1–5.",
      categories: [
        {
          icon: Brain, name: "AI — Week 1–2: Detection Basics", color: "text-purple-600", tasks: [
            { id: "m1-ai-1", text: "Install YOLOv8 via Ultralytics, run on a sample video", resource: "pip install ultralytics → yolo predict" },
            { id: "m1-ai-2", text: "Understand YOLOv8 output format: boxes, scores, class IDs", resource: "Ultralytics docs" },
            { id: "m1-ai-3", text: "Run on your own video (street cam, phone footage, YouTube clip)", resource: "OpenCV VideoCapture" },
            { id: "m1-ai-4", text: "Read YOLOv9 paper abstract + intro — understand PGI/GELAN conceptually", resource: "arxiv.org/abs/2402.13616" },
          ]
        },
        {
          icon: Brain, name: "AI — Week 3–4: Add Tracking", color: "text-indigo-600", tasks: [
            { id: "m1-ai-5", text: "Clone boxmot, integrate DeepSORT with your YOLOv8 output", resource: "github: mikel-brostrom/boxmot" },
            { id: "m1-ai-6", text: "Read SORT paper (6 pages) — Kalman filter + Hungarian matching", resource: "arxiv.org/abs/1602.00763" },
            { id: "m1-ai-7", text: "Get stable track IDs showing on video with bounding boxes", resource: "OpenCV putText, rectangle" },
            { id: "m1-ai-8", text: "Push to GitHub with README and a demo GIF", resource: "→ Your first portfolio piece" },
            { id: "m1-ai-9", text: "Understand MOT metrics: MOTA = tracking accuracy, IDF1 = ID consistency", resource: "MOTChallenge website" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — 3A Chapters 1–5 (Fri + weekday evenings)", color: "text-pink-600", tasks: [
            { id: "m1-kr-1", text: "1과: 신입생 환영회를 한다고 해요 — study + vocab list to Anki", resource: "간접화법: ~다고 해요 / ~(으)라고 해요" },
            { id: "m1-kr-2", text: "New words from 1과 — add ALL new vocab to Anki deck, review daily", resource: "Target: 15-20 new words per chapter" },
            { id: "m1-kr-3", text: "2과: 방을 바꿔 달라고 해 봐 — study + vocab to Anki", resource: "~아/어 달라고 하다, ~(으)라고 하다" },
            { id: "m1-kr-4", text: "New words from 2과 — add to Anki, keep daily review streak", resource: "Review previous chapter cards too" },
            { id: "m1-kr-5", text: "3과: 비가 이렇게 많이 올 줄 몰랐어요 — study + vocab to Anki", resource: "~(으)ㄹ 줄 몰랐다 / 알았다" },
            { id: "m1-kr-6", text: "New words from 3과 — Anki + write 3 example sentences", resource: "Writing sentences = retention booster" },
            { id: "m1-kr-7", text: "4과: 먹어 보니까 맛있던데요 — study + vocab to Anki", resource: "~던데요, ~아/어 보니까" },
            { id: "m1-kr-8", text: "New words from 4과 — Anki + speak them aloud (commute practice)", resource: "Use 서울대 audio files on work commute" },
            { id: "m1-kr-9", text: "5과: 입어 보고 살걸 그랬어요 — study + vocab to Anki", resource: "~(으)ㄹ걸 그랬다 — regret expression" },
            { id: "m1-kr-10", text: "New words from 5과 — Anki + end of month review all 1–5과 grammar", resource: "Quick self-test: can you use each grammar point?" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — Arrays & Hashing + Two Pointers (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m1-lc-1", text: "Two Sum", resource: "Easy - hashmap O(n)" },
            { id: "m1-lc-2", text: "Contains Duplicate", resource: "Easy - set" },
            { id: "m1-lc-3", text: "Valid Anagram", resource: "Easy - char count" },
            { id: "m1-lc-4", text: "Group Anagrams", resource: "Medium - sorted key hashmap" },
            { id: "m1-lc-5", text: "Top K Frequent Elements", resource: "Medium - bucket sort" },
            { id: "m1-lc-6", text: "Product of Array Except Self", resource: "Medium - prefix/suffix arrays" },
            { id: "m1-lc-7", text: "Encode and Decode Strings", resource: "Medium - length prefix encoding" },
            { id: "m1-lc-8", text: "Longest Consecutive Sequence", resource: "Medium - set O(n)" },
            { id: "m1-lc-9", text: "Two Sum II", resource: "Medium - two pointers on sorted array" },
            { id: "m1-lc-10", text: "3Sum", resource: "Medium - sort + two pointers" },
            { id: "m1-lc-11", text: "Valid Palindrome", resource: "Easy - two pointers" },
            { id: "m1-lc-12", text: "Container With Most Water", resource: "Medium - shrink from both ends" },
            { id: "m1-lc-13", text: "Trapping Rain Water", resource: "Hard - prefix max arrays" },
          ]
        },
        {
          icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
            { id: "m1-fin-1", text: "Track expenses, set savings target: $500-750/month", resource: "Automate transfers" },
            { id: "m1-fin-2", text: "Save first installment toward $3,000-4,000 emergency fund", resource: "High-yield savings" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m1-h1", text: "Establish 3x/week exercise (Tue/Wed/Thu — work days are off limits)", resource: "30-45 min, any activity" },
          ]
        },
      ]
    },
    {
      month: 2, title: "ByteTrack + Video Pipeline + Finish 3A", date: "April 2026",
      badge: "📦 First real project", badgeColor: "bg-blue-100 text-blue-700",
      goal: "AI: ByteTrack on MOT17 with logged metrics. Korean: Finish 3A (6–9과) + start Anki maintenance mode.",
      categories: [
        {
          icon: Brain, name: "AI — Week 1–2: ByteTrack Deep Dive", color: "text-purple-600", tasks: [
            { id: "m2-ai-1", text: "Read ByteTrack paper fully — short and practical", resource: "arxiv.org/abs/2110.06864" },
            { id: "m2-ai-2", text: "Swap DeepSORT for ByteTrack in your pipeline using boxmot", resource: "github: mikel-brostrom/boxmot" },
            { id: "m2-ai-3", text: "Download MOT17 dataset (or subset), run your tracker on it", resource: "MOTChallenge download page" },
            { id: "m2-ai-4", text: "Log MOTA and IDF1 scores — understand what moves the numbers", resource: "py-motmetrics library" },
          ]
        },
        {
          icon: Brain, name: "AI — Week 3–4: Video Pipeline", color: "text-indigo-600", tasks: [
            { id: "m2-ai-5", text: "Build clean pipeline: read → detect → track → annotate → write", resource: "OpenCV VideoWriter, FFmpeg" },
            { id: "m2-ai-6", text: "Handle edge cases: empty frames, no detections, video end", resource: "Basic error handling" },
            { id: "m2-ai-7", text: "Choose use case: traffic OR retail/people counting (pick ONE)", resource: "Traffic = easier data, retail = more practical" },
            { id: "m2-ai-8", text: "Draw trajectory lines behind tracked objects", resource: "cv2.polylines with deque history" },
            { id: "m2-ai-9", text: "Update GitHub: clean structure, MOT17 benchmark results in README", resource: "→ Recruiters look at READMEs" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — Finish 3A (6–9과)", color: "text-pink-600", tasks: [
            { id: "m2-kr-1", text: "6과: 일요일에는 아무 약속도 없어요 — study + vocab to Anki", resource: "아무 + negative, ~(이)든지" },
            { id: "m2-kr-2", text: "New words from 6과 — Anki + use in 3 written sentences", resource: "Daily Anki review non-negotiable" },
            { id: "m2-kr-3", text: "7과: 껐다가 다시 켜 보세요 — study + vocab to Anki", resource: "~았/었다가 — sequential then contrast" },
            { id: "m2-kr-4", text: "New words from 7과 — Anki + speak aloud on work commute", resource: "서울대 audio track for 7과" },
            { id: "m2-kr-5", text: "8과: 교통사고가 났다고요? — study + vocab to Anki", resource: "~다고요? surprised echo + reported speech" },
            { id: "m2-kr-6", text: "New words from 8과 — Anki + write a short paragraph using this grammar", resource: "2-3 sentences is enough" },
            { id: "m2-kr-7", text: "9과: 한글날에 대해 들어 봤어요? — study + vocab to Anki", resource: "~에 대해(서), ~(으)ㄹ 뿐만 아니라" },
            { id: "m2-kr-8", text: "New words from 9과 — final Anki batch for 3A", resource: "You now have all 3A vocab in your deck" },
            { id: "m2-kr-9", text: "Full 3A grammar review — self-test all 9 grammar points without notes", resource: "Write one sentence for each grammar pattern" },
            { id: "m2-kr-10", text: "3A complete 🎉 — keep Anki reviews going daily (10 min max)", resource: "Don't let the deck pile up — review every day" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — Sliding Window + Stack + Binary Search (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m2-lc-1", text: "Best Time to Buy and Sell Stock", resource: "Easy - sliding min" },
            { id: "m2-lc-2", text: "Longest Substring Without Repeating Characters", resource: "Medium - sliding window + set" },
            { id: "m2-lc-3", text: "Longest Repeating Character Replacement", resource: "Medium - window + max freq" },
            { id: "m2-lc-4", text: "Permutation in String", resource: "Medium - fixed window char count" },
            { id: "m2-lc-5", text: "Minimum Window Substring", resource: "Hard - shrink window" },
            { id: "m2-lc-6", text: "Sliding Window Maximum", resource: "Hard - monotonic deque" },
            { id: "m2-lc-7", text: "Valid Parentheses", resource: "Easy - stack matching" },
            { id: "m2-lc-8", text: "Min Stack", resource: "Medium - parallel min stack" },
            { id: "m2-lc-9", text: "Evaluate Reverse Polish Notation", resource: "Medium - stack ops" },
            { id: "m2-lc-10", text: "Generate Parentheses", resource: "Medium - backtrack open/close counts" },
            { id: "m2-lc-11", text: "Daily Temperatures", resource: "Medium - monotonic decreasing stack" },
            { id: "m2-lc-12", text: "Car Fleet", resource: "Medium - stack + sorting" },
            { id: "m2-lc-13", text: "Largest Rectangle in Histogram", resource: "Hard - monotonic stack" },
            { id: "m2-lc-14", text: "Binary Search", resource: "Easy - classic template" },
            { id: "m2-lc-15", text: "Search in Rotated Sorted Array", resource: "Medium - determine sorted half" },
            { id: "m2-lc-16", text: "Find Minimum in Rotated Sorted Array", resource: "Medium - binary search boundary" },
            { id: "m2-lc-17", text: "Search a 2D Matrix", resource: "Medium - treat as 1D array" },
            { id: "m2-lc-18", text: "Koko Eating Bananas", resource: "Medium - binary search on answer" },
          ]
        },
        {
          icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
            { id: "m2-fin-1", text: "Save second installment (total: $1,000-1,500)", resource: "Stay consistent" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m2-h1", text: "Continue 3x/week routine, add variety", resource: "Mix cardio + strength" },
          ]
        },
      ]
    },
    {
      month: 3, title: "Analytics + API + Start 3B", date: "May 2026",
      badge: "📊 Something to demo", badgeColor: "bg-purple-100 text-purple-700",
      goal: "AI: Working analytics + /upload API. Korean: 3B chapters 1–5 + Anki streak maintained.",
      categories: [
        {
          icon: Brain, name: "AI — Week 1–2: Domain Analytics", color: "text-purple-600", tasks: [
            { id: "m3-ai-1", text: "Add object counter: total counts per class per video", resource: "Simple dict counter" },
            { id: "m3-ai-2", text: "Add zone-crossing detection: draw a line, count objects that cross it", resource: "Line intersection math" },
            { id: "m3-ai-3", text: "Generate trajectory heatmap as output image", resource: "numpy + matplotlib imshow" },
            { id: "m3-ai-4", text: "Export results as JSON: {track_id, class, frames_seen, trajectory}", resource: "json.dump" },
          ]
        },
        {
          icon: Briefcase, name: "AI — Week 3–4: Basic API", color: "text-blue-600", tasks: [
            { id: "m3-ai-5", text: "Build FastAPI with POST /upload and GET /results/{job_id}", resource: "fastapi.tiangolo.com" },
            { id: "m3-ai-6", text: "Process video in background (simple threading, no Celery yet)", resource: "BackgroundTasks in FastAPI" },
            { id: "m3-ai-7", text: "Profile pipeline with cProfile — find #1 bottleneck and fix it", resource: "python -m cProfile" },
            { id: "m3-ai-8", text: "Record 2-3 min demo video of full pipeline in action", resource: "OBS or Loom — for portfolio" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — 3B Chapters 1–5", color: "text-pink-600", tasks: [
            { id: "m3-kr-1", text: "3B 1과 — study grammar point, add new vocab to Anki", resource: "New grammar pattern + 15-20 words" },
            { id: "m3-kr-2", text: "New words 3B 1과 — Anki + 3 example sentences", resource: "Keep the writing habit" },
            { id: "m3-kr-3", text: "3B 2과 — study grammar point, add new vocab to Anki", resource: "Build on 3A patterns" },
            { id: "m3-kr-4", text: "New words 3B 2과 — Anki + speak on commute", resource: "서울대 3B audio files" },
            { id: "m3-kr-5", text: "3B 3과 — study grammar point, add new vocab to Anki", resource: "Check: are you actually using these?" },
            { id: "m3-kr-6", text: "New words 3B 3과 — Anki + write short paragraph", resource: "Paragraph practice builds fluency" },
            { id: "m3-kr-7", text: "3B 4과 — study grammar point, add new vocab to Anki", resource: "Mid-month grammar check" },
            { id: "m3-kr-8", text: "New words 3B 4과 — Anki daily review streak", resource: "Don't miss a single day" },
            { id: "m3-kr-9", text: "3B 5과 — study grammar point, add new vocab to Anki", resource: "Halfway through 3B 🎉" },
            { id: "m3-kr-10", text: "New words 3B 5과 + review all 3B grammar so far (1–5과)", resource: "Self-test: one sentence each without notes" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — Linked List + Trees (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m3-lc-1", text: "Reverse Linked List", resource: "Easy - iterative prev/curr" },
            { id: "m3-lc-2", text: "Merge Two Sorted Lists", resource: "Easy - pointer merge" },
            { id: "m3-lc-3", text: "Linked List Cycle", resource: "Easy - fast/slow pointers" },
            { id: "m3-lc-4", text: "Reorder List", resource: "Medium - find mid + reverse + merge" },
            { id: "m3-lc-5", text: "Remove Nth Node From End", resource: "Medium - two pointers gap n" },
            { id: "m3-lc-6", text: "Find the Duplicate Number", resource: "Medium - Floyd's cycle detection" },
            { id: "m3-lc-7", text: "LRU Cache", resource: "Medium - hashmap + doubly linked list" },
            { id: "m3-lc-8", text: "Merge K Sorted Lists", resource: "Hard - min-heap" },
            { id: "m3-lc-9", text: "Reverse Nodes in K-Group", resource: "Hard - recursive group reverse" },
            { id: "m3-lc-10", text: "Invert Binary Tree", resource: "Easy - recursive swap" },
            { id: "m3-lc-11", text: "Maximum Depth of Binary Tree", resource: "Easy - DFS" },
            { id: "m3-lc-12", text: "Diameter of Binary Tree", resource: "Easy - depth + max at each node" },
            { id: "m3-lc-13", text: "Balanced Binary Tree", resource: "Easy - height check recursive" },
            { id: "m3-lc-14", text: "Same Tree", resource: "Easy - recursive compare" },
            { id: "m3-lc-15", text: "Subtree of Another Tree", resource: "Easy - isSameTree helper" },
            { id: "m3-lc-16", text: "Lowest Common Ancestor of BST", resource: "Medium - compare vals to root" },
            { id: "m3-lc-17", text: "Binary Tree Level Order Traversal", resource: "Medium - BFS queue" },
            { id: "m3-lc-18", text: "Binary Tree Right Side View", resource: "Medium - BFS last of each level" },
          ]
        },
        {
          icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
            { id: "m3-fin-1", text: "Save third installment (total: $1,500-2,250)", resource: "Halfway there!" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m3-h1", text: "Maintain 3x/week routine", resource: "Build the habit" },
          ]
        },
      ]
    },
    {
      month: 4, title: "Deploy + Finish 3B + Start Applying", date: "June 2026",
      badge: "🌐 Live demo + job hunt starts", badgeColor: "bg-orange-100 text-orange-700",
      goal: "AI: Live URL deployed. Korean: Finish 3B (6–9과). Applications: 10+ sent.",
      categories: [
        {
          icon: Briefcase, name: "AI — Week 1–2: Containerize", color: "text-blue-600", tasks: [
            { id: "m4-ai-1", text: "Write a Dockerfile for your FastAPI + tracker app", resource: "Multi-stage build to keep image small" },
            { id: "m4-ai-2", text: "Set up docker-compose for local dev", resource: "docker-compose.yml" },
            { id: "m4-ai-3", text: "Add structured logging and basic error handling", resource: "Python logging module" },
            { id: "m4-ai-4", text: "Auto-generate API docs via FastAPI Swagger UI (/docs)", resource: "Free and impressive to show" },
          ]
        },
        {
          icon: Briefcase, name: "AI — Week 3–4: Deploy & Apply", color: "text-indigo-600", tasks: [
            { id: "m4-ai-5", text: "Deploy to Railway, Render, or EC2 — get a live URL", resource: "Railway is easiest; EC2 gives more CV cred" },
            { id: "m4-ai-6", text: "Add GitHub Actions: auto-deploy on push to main", resource: "CI/CD basics" },
            { id: "m4-ai-7", text: "Build simple Streamlit dashboard: counts, heatmap, live stats", resource: "streamlit.io" },
            { id: "m4-ai-8", text: "🎯 START APPLYING: AI/CV/ML internships for Autumn — 10+ companies", resource: "LinkedIn, Glassdoor, company career pages" },
            { id: "m4-ai-9", text: "Polish LinkedIn: live URL, demo GIF, benchmark numbers", resource: "Recruiters will Google your name" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — Finish 3B (6–9과)", color: "text-pink-600", tasks: [
            { id: "m4-kr-1", text: "3B 6과 — study grammar point, add new vocab to Anki", resource: "Two-thirds through 3B" },
            { id: "m4-kr-2", text: "New words 3B 6과 — Anki + sentences", resource: "Keep writing practice going" },
            { id: "m4-kr-3", text: "3B 7과 — study grammar point, add new vocab to Anki", resource: "Audio on work commute" },
            { id: "m4-kr-4", text: "New words 3B 7과 — Anki daily", resource: "Non-negotiable 10 min" },
            { id: "m4-kr-5", text: "3B 8과 — study grammar point, add new vocab to Anki", resource: "Almost there" },
            { id: "m4-kr-6", text: "New words 3B 8과 — Anki + paragraph practice", resource: "Write 4-5 sentences using this chapter's grammar" },
            { id: "m4-kr-7", text: "3B 9과 — study grammar point, add new vocab to Anki", resource: "Final chapter of 3B 🎉" },
            { id: "m4-kr-8", text: "New words 3B 9과 — complete Anki deck for all of 3B", resource: "You now have 3A + 3B fully in Anki" },
            { id: "m4-kr-9", text: "Full 3B grammar review — self-test all 9 grammar points", resource: "One sentence each from memory" },
            { id: "m4-kr-10", text: "3B complete 🎉 — decide next step: 4A or TOPIK II prep", resource: "You've covered intermediate grammar solidly" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — Trees (cont.) + Tries + Heap + Backtracking (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m4-lc-1", text: "Count Good Nodes in Binary Tree", resource: "Medium - DFS max so far" },
            { id: "m4-lc-2", text: "Validate Binary Search Tree", resource: "Medium - DFS with min/max bounds" },
            { id: "m4-lc-3", text: "Kth Smallest Element in BST", resource: "Medium - inorder traversal" },
            { id: "m4-lc-4", text: "Construct BST from Preorder Traversal", resource: "Medium - recursion with bounds" },
            { id: "m4-lc-5", text: "Binary Tree Maximum Path Sum", resource: "Hard - DFS gain at each node" },
            { id: "m4-lc-6", text: "Serialize and Deserialize Binary Tree", resource: "Hard - BFS or preorder with null markers" },
            { id: "m4-lc-7", text: "Implement Trie (Prefix Tree)", resource: "Medium - TrieNode children dict" },
            { id: "m4-lc-8", text: "Add and Search Word", resource: "Medium - DFS with wildcard dot" },
            { id: "m4-lc-9", text: "Word Search II", resource: "Hard - Trie + DFS backtrack on board" },
            { id: "m4-lc-10", text: "Kth Largest Element in Array", resource: "Medium - min-heap size k" },
            { id: "m4-lc-11", text: "K Closest Points to Origin", resource: "Medium - max-heap or quickselect" },
            { id: "m4-lc-12", text: "Task Scheduler", resource: "Medium - greedy max freq" },
            { id: "m4-lc-13", text: "Design Twitter", resource: "Medium - heap + user follow map" },
            { id: "m4-lc-14", text: "Find Median from Data Stream", resource: "Hard - two heaps (max + min)" },
            { id: "m4-lc-15", text: "Subsets", resource: "Medium - backtrack include/exclude" },
            { id: "m4-lc-16", text: "Combination Sum", resource: "Medium - backtrack with repeat" },
            { id: "m4-lc-17", text: "Permutations", resource: "Medium - backtrack swap" },
            { id: "m4-lc-18", text: "Word Search", resource: "Medium - DFS + visited in-place" },
            { id: "m4-lc-19", text: "N-Queens", resource: "Hard - backtrack with col/diag sets" },
          ]
        },
        {
          icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
            { id: "m4-fin-1", text: "Save fourth installment (total: $2,000-3,000)", resource: "Getting close!" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m4-h1", text: "Exercise 3-4x/week — should feel natural now", resource: "Consistency matters" },
          ]
        },
      ]
    },
    {
      month: 5, title: "Polish + Interview Prep + Korean Consolidation", date: "July 2026",
      badge: "🎯 Interview season", badgeColor: "bg-red-100 text-red-700",
      goal: "AI: Interview-ready, blog post live. Korean: Anki maintenance + start reading Korean tech content.",
      categories: [
        {
          icon: Brain, name: "AI — Week 1–2: Deepen Technical Knowledge", color: "text-purple-600", tasks: [
            { id: "m5-ai-1", text: "Read DeepSORT paper fully — appearance features + Re-ID", resource: "arxiv.org/abs/1703.07402" },
            { id: "m5-ai-2", text: "Add TensorRT or ONNX export to pipeline (speed boost)", resource: "Ultralytics export guide" },
            { id: "m5-ai-3", text: "Try BoT-SORT or StrongSORT — compare metrics vs ByteTrack", resource: "boxmot tracker swap" },
            { id: "m5-ai-4", text: "Add MLflow: log FPS, MOTA, IDF1 per experiment run", resource: "mlflow.org" },
          ]
        },
        {
          icon: Briefcase, name: "AI — Week 3–4: Interview Prep", color: "text-blue-600", tasks: [
            { id: "m5-ai-5", text: "Write technical blog post: 'How I built a real-time MOT system'", resource: "Medium or Dev.to — shows communication skills" },
            { id: "m5-ai-6", text: "Practice explaining your project in 2 min (record yourself)", resource: "Watch it back — painful but effective" },
            { id: "m5-ai-7", text: "Prep CV interview questions: detection, tracking, Kalman, Hungarian algorithm", resource: "Papers With Code MOT section" },
            { id: "m5-ai-9", text: "Apply to 20+ more companies — prioritize startups, they move faster", resource: "YC companies, LinkedIn Easy Apply" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — Consolidation & Real Usage", color: "text-pink-600", tasks: [
            { id: "m5-kr-1", text: "Daily Anki reviews — 10 min max, keep the streak alive", resource: "All 3A + 3B vocab = ~200-250 cards by now" },
            { id: "m5-kr-2", text: "Read one short Korean article or webtoon per week", resource: "네이버 뉴스 easy articles, 한국어 webtoons" },
            { id: "m5-kr-3", text: "Write one short Korean paragraph per week (diary style)", resource: "오늘 뭐 했어요? Use grammar from 3A/3B" },
            { id: "m5-kr-4", text: "Listen to Korean podcast or YouTube 20 min on work commute (3x/week)", resource: "TTMIK, 이상한 나라의 며느리, Korean Unnie" },
            { id: "m5-kr-5", text: "Decide: start 4A in Aug OR focus on TOPIK II reading practice", resource: "TOPIK II = more useful for Korean job market" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — Graphs + Dynamic Programming Part 1 (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m5-lc-1", text: "Number of Islands", resource: "Medium - DFS/BFS flood fill" },
            { id: "m5-lc-2", text: "Max Area of Island", resource: "Medium - DFS return area count" },
            { id: "m5-lc-3", text: "Clone Graph", resource: "Medium - BFS + hashmap visited" },
            { id: "m5-lc-4", text: "Walls and Gates", resource: "Medium - multi-source BFS" },
            { id: "m5-lc-5", text: "Rotting Oranges", resource: "Medium - multi-source BFS time" },
            { id: "m5-lc-6", text: "Pacific Atlantic Water Flow", resource: "Medium - reverse BFS from both coasts" },
            { id: "m5-lc-7", text: "Surrounded Regions", resource: "Medium - DFS from borders" },
            { id: "m5-lc-8", text: "Course Schedule", resource: "Medium - cycle detection DFS" },
            { id: "m5-lc-9", text: "Course Schedule II", resource: "Medium - topological sort" },
            { id: "m5-lc-10", text: "Number of Connected Components", resource: "Medium - Union-Find or DFS" },
            { id: "m5-lc-11", text: "Redundant Connection", resource: "Medium - Union-Find detect cycle" },
            { id: "m5-lc-12", text: "Word Ladder", resource: "Hard - BFS shortest path" },
            { id: "m5-lc-13", text: "Climbing Stairs", resource: "Easy - DP fib pattern" },
            { id: "m5-lc-14", text: "Min Cost Climbing Stairs", resource: "Easy - DP bottom-up" },
            { id: "m5-lc-15", text: "House Robber", resource: "Medium - DP no adjacent" },
            { id: "m5-lc-16", text: "House Robber II", resource: "Medium - two passes circular" },
            { id: "m5-lc-17", text: "Longest Palindromic Substring", resource: "Medium - expand around center" },
            { id: "m5-lc-18", text: "Coin Change", resource: "Medium - DP bottom-up BFS" },
            { id: "m5-lc-19", text: "Word Break", resource: "Medium - DP + set lookup" },
            { id: "m5-lc-20", text: "Longest Increasing Subsequence", resource: "Medium - DP O(n2) or patience sort" },
          ]
        },
        {
          icon: DollarSign, name: "Financial", color: "text-green-600", tasks: [
            { id: "m5-fin-1", text: "Save fifth installment (total: $2,500-3,750)", resource: "Almost there!" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m5-h1", text: "Maintain routine — it's a lifestyle now", resource: "Keep it up" },
          ]
        },
      ]
    },
    {
      month: 6, title: "Launch, Contribute & Land the Internship", date: "August 2026",
      badge: "🏆 Finish strong", badgeColor: "bg-yellow-100 text-yellow-700",
      goal: "AI: Internship offer. Korean: Consistent daily practice, real content consumption habit built.",
      categories: [
        {
          icon: Rocket, name: "AI — Week 1–2: Open Source + Advanced Features", color: "text-purple-600", tasks: [
            { id: "m6-ai-1", text: "Submit one PR to Ultralytics or roboflow/supervision (docs/bug fix counts)", resource: "Shows collaboration — gold for recruiters" },
            { id: "m6-ai-2", text: "Add one advanced feature: pose estimation OR multi-camera ReID", resource: "YOLOv8-pose or boxmot cross-camera" },
            { id: "m6-ai-3", text: "Add Prometheus metrics + simple Grafana dashboard for latency/FPS", resource: "Shows MLOps awareness" },
          ]
        },
        {
          icon: Briefcase, name: "AI — Week 3–4: Land It", color: "text-blue-600", tasks: [
            { id: "m6-ai-4", text: "Final GitHub polish: README, architecture diagram, demo video, badges", resource: "shields.io for badges" },
            { id: "m6-ai-5", text: "Create case study: FPS, MOTA, cost to run, use case impact", resource: "One-page PDF or Notion page" },
            { id: "m6-ai-6", text: "Share on LinkedIn + r/computervision + r/MachineLearning", resource: "People genuinely get hired from this" },
            { id: "m6-ai-7", text: "Follow up all pending apps. Cold email CV/ML engineers at target companies", resource: "5-line email: who you are, what you built, link, ask 15 min" },
            { id: "m6-ai-8", text: "Read one new MOT/CV ArXiv paper per week", resource: "Stay sharp for final round interviews" },
          ]
        },
        {
          icon: BookOpen, name: "한국어 — Maintain & Push Forward", color: "text-pink-600", tasks: [
            { id: "m6-kr-1", text: "Daily Anki reviews — non-negotiable 10 min", resource: "Consistency beats intensity" },
            { id: "m6-kr-2", text: "Start 4A chapter 1 OR begin TOPIK II past paper practice (based on May decision)", resource: "Either path is progress" },
            { id: "m6-kr-3", text: "Read 2 Korean articles this month — look up unknown words, add to Anki", resource: "네이버 뉴스 or 한겨레 easy section" },
            { id: "m6-kr-4", text: "Write 4 short Korean diary entries this month (one per week)", resource: "주간 일기 — document your internship hunt in Korean!" },
            { id: "m6-kr-5", text: "Reflect: how much Korean can you actually use now vs March?", resource: "You've covered 18 chapters in 6 months 💪" },
          ]
        },
        {
          icon: Code2, name: "LeetCode — DP Part 2 + Greedy + Intervals + Bit Manipulation (Weeks 1–4)", color: "text-orange-600", tasks: [
            { id: "m6-lc-1", text: "Partition Equal Subset Sum", resource: "Medium - 0/1 knapsack DP" },
            { id: "m6-lc-2", text: "Unique Paths", resource: "Medium - DP grid" },
            { id: "m6-lc-3", text: "Jump Game", resource: "Medium - greedy max reach" },
            { id: "m6-lc-4", text: "Jump Game II", resource: "Medium - greedy BFS levels" },
            { id: "m6-lc-5", text: "Gas Station", resource: "Medium - greedy prefix sum" },
            { id: "m6-lc-6", text: "Hand of Straights", resource: "Medium - greedy + sorted map" },
            { id: "m6-lc-7", text: "Edit Distance", resource: "Medium - 2D DP classic" },
            { id: "m6-lc-8", text: "Burst Balloons", resource: "Hard - interval DP" },
            { id: "m6-lc-9", text: "Regular Expression Matching", resource: "Hard - 2D DP with dot and star" },
            { id: "m6-lc-10", text: "Insert Interval", resource: "Medium - linear merge" },
            { id: "m6-lc-11", text: "Merge Intervals", resource: "Medium - sort + merge" },
            { id: "m6-lc-12", text: "Non-overlapping Intervals", resource: "Medium - greedy min removals" },
            { id: "m6-lc-13", text: "Meeting Rooms", resource: "Easy - sort + overlap check" },
            { id: "m6-lc-14", text: "Meeting Rooms II", resource: "Medium - min-heap end times" },
            { id: "m6-lc-15", text: "Minimum Interval to Include Each Query", resource: "Hard - sweep + heap" },
            { id: "m6-lc-16", text: "Single Number", resource: "Easy - XOR all elements" },
            { id: "m6-lc-17", text: "Number of 1 Bits", resource: "Easy - n and (n-1) trick" },
            { id: "m6-lc-18", text: "Counting Bits", resource: "Easy - DP i right-shift 1" },
            { id: "m6-lc-19", text: "Reverse Bits", resource: "Easy - shift and OR" },
            { id: "m6-lc-20", text: "Missing Number", resource: "Easy - XOR 0 to n" },
            { id: "m6-lc-21", text: "Sum of Two Integers", resource: "Medium - bit carry simulation" },
            { id: "m6-lc-22", text: "Reverse Integer", resource: "Medium - pop and push digits" },
          ]
        },
        {
          icon: Star, name: "Continuous Learning", color: "text-yellow-600", tasks: [
            { id: "m6-l1", text: "Contribute to open source: Ultralytics or supervision library", resource: "github.com/ultralytics, roboflow/supervision" },
          ]
        },
        {
          icon: DollarSign, name: "Financial — Goal Achieved! 🎉", color: "text-green-600", tasks: [
            { id: "m6-fin-1", text: "Final savings push (total: $3,000-4,000+ emergency fund!)", resource: "Mission accomplished!" },
            { id: "m6-fin-2", text: "Plan next financial milestone: 6-month fund or investments", resource: "Continue growing" },
          ]
        },
        {
          icon: Dumbbell, name: "Health", color: "text-red-600", tasks: [
            { id: "m6-h1", text: "Celebrate health gains — you've built a real habit", resource: "Set the next fitness challenge" },
          ]
        },
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

        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 md:p-8 mb-6">
          <div className="flex items-center gap-3 mb-2">
            <Target className="w-8 h-8 text-indigo-600" />
            <h1 className="text-3xl md:text-4xl font-bold text-slate-900">6-Month Realistic Plan</h1>
          </div>
          <p className="text-slate-500 mb-1 text-sm">March – August 2026 · Student + 36hr work week · 1-2 hrs/day · Autumn AI internship</p>
          <p className="text-indigo-600 font-medium text-sm mb-4">AI project + 서울대 한국어 3A→3B + LeetCode 150 + job hunt. Each month = one thing to show a recruiter.</p>
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

        {/* Weekly Schedule */}
        <div className="bg-white rounded-2xl shadow-lg overflow-hidden mb-4">
          <button onClick={() => setExpandedSchedule(p => !p)} className="w-full p-6 flex items-center justify-between hover:bg-slate-50 transition-colors">
            <div className="flex items-center gap-3">
              <Calendar className="w-6 h-6 text-indigo-600" />
              <div className="text-left">
                <h2 className="text-lg font-bold text-slate-900">📅 Your Realistic Weekly Schedule</h2>
                <p className="text-sm text-slate-500">Based on your actual life — expand to see</p>
              </div>
            </div>
            {expandedSchedule ? <ChevronDown className="w-6 h-6 text-slate-400" /> : <ChevronRight className="w-6 h-6 text-slate-400" />}
          </button>
          {expandedSchedule && (
            <div className="px-6 pb-6 grid md:grid-cols-3 gap-4">
              {weeklySchedule.map((s, i) => (
                <div key={i} className={`rounded-xl border p-4 ${s.color}`}>
                  <div className="font-bold text-slate-800 text-sm mb-1">{s.day}</div>
                  <div className="text-xs text-slate-500 mb-3 font-medium uppercase tracking-wide">{s.label}</div>
                  <ul className="space-y-2">
                    {s.items.map((item, j) => (
                      <li key={j} className="text-sm text-slate-700 flex gap-2">
                        <span className="mt-0.5 text-slate-400">•</span>
                        <span>{item}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
              <div className="md:col-span-3 bg-amber-50 border border-amber-200 rounded-xl p-4">
                <p className="text-sm text-amber-800 font-medium">⚠️ Honest note: University courses are ranked 3rd in your priorities but failing them hurts your internship applications too. When exam season hits, let the AI project slow down — not your grades.</p>
              </div>
            </div>
          )}
        </div>

        {/* Monthly Roadmap */}
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

        {/* Resources */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mt-6">
          <h3 className="text-xl font-bold text-slate-900 mb-4">🔗 Essential Resources</h3>
          <div className="grid md:grid-cols-4 gap-4 text-sm">
            <div>
              <h4 className="font-semibold text-purple-600 mb-2">Core AI/Tracking</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• boxmot — plug-and-play trackers</li>
                <li>• ByteTrack paper (2021)</li>
                <li>• SORT paper (2016) — 6 pages only</li>
                <li>• MOTChallenge benchmark</li>
                <li>• py-motmetrics</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-blue-600 mb-2">Production Stack</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• FastAPI + Streamlit</li>
                <li>• Docker + Railway/Render</li>
                <li>• MLflow experiments</li>
                <li>• GitHub Actions CI/CD</li>
                <li>• shields.io for README badges</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-orange-600 mb-2">LeetCode 150</h4>
              <ul className="space-y-1 text-slate-600">
                <li>M1: Arrays, Hashing, Two Pointers</li>
                <li>M2: Sliding Window, Stack, Binary Search</li>
                <li>M3: Linked List, Trees</li>
                <li>M4: Tries, Heap, Backtracking</li>
                <li>M5: Graphs + DP Part 1</li>
                <li>M6: DP Part 2, Greedy, Intervals, Bits</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold text-pink-600 mb-2">한국어 Resources</h4>
              <ul className="space-y-1 text-slate-600">
                <li>• 서울대 한국어 3A/3B textbook + audio</li>
                <li>• Anki (free flashcard app)</li>
                <li>• TTMIK podcast (commute)</li>
                <li>• 네이버 뉴스 — easy Korean reading</li>
                <li>• Korean Unnie YouTube</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="text-center text-slate-500 text-sm mt-8 pb-4">
          <p>Progress saves automatically. Open M1 and start checking boxes <strong>today</strong>.</p>
          <p className="mt-1">🏁 One working demo beats ten half-read papers. 한 걸음씩 — one step at a time.</p>
        </div>
      </div>
    </div>
  );
}
