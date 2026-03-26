import React, { useState, useEffect, useRef } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { doc, getDoc, setDoc, serverTimestamp } from 'firebase/firestore'
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage'
import { db, storage } from '../firebase'
import { PROBLEMS } from '../data/problems'

const DIFF_COLOR = { Easy:'#00ff88', Medium:'#ffb800', Hard:'#ff4444' }
const LANGS = ['python', 'javascript', 'java', 'cpp', 'typescript', 'go']

export default function ProblemEditor() {
  const { problemId } = useParams()
  const navigate = useNavigate()
  const problem = PROBLEMS.find(p => p.id === problemId)
  const fileInputRef = useRef()

  const [form, setForm] = useState({
    status: 'in-progress',
    language: 'python',
    code: '',
    notes: '',
    timeComplexity: '',
    spaceComplexity: '',
    approach: '',
    files: []
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [uploading, setUploading] = useState(false)

  useEffect(() => {
    getDoc(doc(db, 'solutions', problemId)).then(d => {
      if (d.exists()) setForm(prev => ({ ...prev, ...d.data() }))
    })
  }, [problemId])

  const save = async () => {
    setSaving(true)
    try {
      await setDoc(doc(db, 'solutions', problemId), { ...form, updatedAt: serverTimestamp() }, { merge: true })
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally { setSaving(false) }
  }

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files)
    if (!files.length) return
    setUploading(true)
    try {
      const uploaded = await Promise.all(files.map(async file => {
        const r = ref(storage, `solutions/${problemId}/${file.name}`)
        await uploadBytes(r, file)
        const url = await getDownloadURL(r)
        return { name: file.name, url, size: file.size }
      }))
      setForm(f => ({ ...f, files: [...(f.files || []), ...uploaded] }))
    } finally { setUploading(false) }
  }

  const removeFile = (idx) => setForm(f => ({ ...f, files: f.files.filter((_, i) => i !== idx) }))

  if (!problem) return <div style={{padding:'2rem',color:'#ff4444',fontFamily:'JetBrains Mono'}}>Problem not found</div>

  return (
    <div style={s.page}>
      <header style={s.header}>
        <Link to="/dashboard" style={s.back}>← dashboard</Link>
        <div style={s.headerMid}>
          <span style={{...s.diff, color:DIFF_COLOR[problem.difficulty]}}>{problem.difficulty}</span>
          <span style={s.title}>{problem.title}</span>
          <span style={s.topic}>{problem.topic}</span>
        </div>
        <div style={s.headerRight}>
          <a href={problem.leetcodeUrl} target="_blank" rel="noopener noreferrer" style={s.lcLink}>LeetCode ↗</a>
          <button onClick={save} disabled={saving} style={{...s.saveBtn, opacity: saving?0.7:1}}>
            {saving ? 'saving...' : saved ? '✓ saved!' : 'save'}
          </button>
        </div>
      </header>

      <div style={s.container}>
        {/* Status + Meta row */}
        <div style={s.metaRow}>
          <div style={s.field}>
            <label style={s.label}>status</label>
            <select style={s.select} value={form.status} onChange={e => setForm(f=>({...f,status:e.target.value}))}>
              <option value="unsolved">unsolved</option>
              <option value="in-progress">in progress</option>
              <option value="solved">solved</option>
            </select>
          </div>
          <div style={s.field}>
            <label style={s.label}>language</label>
            <select style={s.select} value={form.language} onChange={e => setForm(f=>({...f,language:e.target.value}))}>
              {LANGS.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <div style={s.field}>
            <label style={s.label}>time complexity</label>
            <input style={s.input} placeholder="O(n)" value={form.timeComplexity} onChange={e => setForm(f=>({...f,timeComplexity:e.target.value}))} />
          </div>
          <div style={s.field}>
            <label style={s.label}>space complexity</label>
            <input style={s.input} placeholder="O(1)" value={form.spaceComplexity} onChange={e => setForm(f=>({...f,spaceComplexity:e.target.value}))} />
          </div>
        </div>

        {/* Two column layout */}
        <div style={s.cols}>
          {/* Left: code */}
          <div style={s.col}>
            <label style={s.label}>solution code</label>
            <textarea style={s.codeArea} value={form.code}
              onChange={e => setForm(f=>({...f,code:e.target.value}))}
              placeholder={`# paste your ${form.language} solution here\n\nclass Solution:\n    def twoSum(self, nums, target):\n        ...`}
              spellCheck={false}
            />
          </div>

          {/* Right: notes */}
          <div style={s.col}>
            <label style={s.label}>approach / explanation</label>
            <textarea style={{...s.codeArea, fontFamily:"'Syne',sans-serif", fontSize:'0.85rem', lineHeight:1.7}}
              value={form.approach}
              onChange={e => setForm(f=>({...f,approach:e.target.value}))}
              placeholder="Explain your approach here. What pattern does this use? Why does it work?"
            />
            <label style={{...s.label, marginTop:'1rem'}}>notes / learnings</label>
            <textarea style={{...s.codeArea, height:'140px', fontFamily:"'Syne',sans-serif", fontSize:'0.85rem', lineHeight:1.7}}
              value={form.notes}
              onChange={e => setForm(f=>({...f,notes:e.target.value}))}
              placeholder="What did you struggle with? What's the key insight? What to remember next time?"
            />
          </div>
        </div>

        {/* File upload */}
        <div style={s.uploadSection}>
          <label style={s.label}>uploaded files</label>
          <div style={s.fileRow}>
            <button onClick={() => fileInputRef.current.click()} style={s.uploadBtn} disabled={uploading}>
              {uploading ? 'uploading...' : '+ upload file (.py, .js, .pdf, image)'}
            </button>
            <input ref={fileInputRef} type="file" multiple style={{display:'none'}} onChange={handleFileUpload}
              accept=".py,.js,.ts,.java,.cpp,.go,.pdf,.png,.jpg,.txt,.md" />
          </div>
          {form.files?.length > 0 && (
            <div style={s.fileList}>
              {form.files.map((f, i) => (
                <div key={i} style={s.fileItem}>
                  <a href={f.url} target="_blank" rel="noopener noreferrer" style={s.fileName}>{f.name}</a>
                  <span style={s.fileSize}>{(f.size/1024).toFixed(1)}kb</span>
                  <button onClick={() => removeFile(i)} style={s.removeFile}>×</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', background:'#0a0a0a', color:'#f0f0f0' },
  header: { display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0.9rem 2rem', borderBottom:'1px solid #1a1a1a', position:'sticky', top:0, background:'#0a0a0a', zIndex:10, flexWrap:'wrap', gap:'0.5rem' },
  back: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#555', textDecoration:'none' },
  headerMid: { display:'flex', alignItems:'center', gap:'0.75rem', flex:1, justifyContent:'center' },
  diff: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', fontWeight:700 },
  title: { fontFamily:"'Syne',sans-serif", fontSize:'1rem', fontWeight:700 },
  topic: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555' },
  headerRight: { display:'flex', gap:'0.75rem', alignItems:'center' },
  lcLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', color:'#555', textDecoration:'none' },
  saveBtn: { background:'#00ff88', color:'#0a0a0a', border:'none', borderRadius:'6px', padding:'0.45rem 1.2rem', fontWeight:700, fontFamily:"'Syne',sans-serif", fontSize:'0.8rem', cursor:'pointer' },
  container: { maxWidth:'1200px', margin:'0 auto', padding:'1.5rem 2rem' },
  metaRow: { display:'flex', gap:'1rem', flexWrap:'wrap', marginBottom:'1.5rem', background:'#111', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem 1.2rem' },
  field: { display:'flex', flexDirection:'column', gap:'0.3rem', minWidth:'140px' },
  label: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', textTransform:'uppercase', letterSpacing:'0.08em' },
  select: { background:'#0a0a0a', border:'1px solid #2a2a2a', borderRadius:'5px', color:'#f0f0f0', padding:'0.4rem 0.6rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.8rem' },
  input: { background:'#0a0a0a', border:'1px solid #2a2a2a', borderRadius:'5px', color:'#f0f0f0', padding:'0.4rem 0.6rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.8rem', outline:'none', width:'120px' },
  cols: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem', marginBottom:'1.5rem' },
  col: { display:'flex', flexDirection:'column', gap:'0.5rem' },
  codeArea: { background:'#0d0d0d', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem', color:'#f0f0f0', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.82rem', lineHeight:1.7, resize:'vertical', minHeight:'340px', outline:'none', tabSize:4 },
  uploadSection: { background:'#111', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem 1.2rem' },
  fileRow: { marginTop:'0.5rem' },
  uploadBtn: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', background:'none', border:'1px dashed #2a2a2a', color:'#888', padding:'0.5rem 1rem', borderRadius:'6px', cursor:'pointer' },
  fileList: { marginTop:'0.75rem', display:'flex', flexDirection:'column', gap:'0.4rem' },
  fileItem: { display:'flex', alignItems:'center', gap:'0.75rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem' },
  fileName: { color:'#00ff88', textDecoration:'none', flex:1 },
  fileSize: { color:'#555' },
  removeFile: { background:'none', border:'none', color:'#ff4444', cursor:'pointer', fontSize:'1rem', lineHeight:1 },
}
