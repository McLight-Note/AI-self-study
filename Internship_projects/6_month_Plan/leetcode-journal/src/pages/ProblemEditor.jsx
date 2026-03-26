import React, { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { doc, getDoc, setDoc, serverTimestamp } from 'firebase/firestore'
import { db } from '../firebase'
import { PROBLEMS } from '../data/problems'

const DIFF_COLOR = { Easy:'#00ff88', Medium:'#ffb800', Hard:'#ff4444' }
const LANGS = ['python', 'javascript', 'java', 'cpp', 'typescript', 'go']

export default function ProblemEditor() {
  const { problemId } = useParams()
  const navigate = useNavigate()
  const problem = PROBLEMS.find(p => p.id === problemId)

  const [form, setForm] = useState({
    status: 'in-progress',
    language: 'python',
    code: '',
    notes: '',
    timeComplexity: '',
    spaceComplexity: '',
    approach: '',
  })
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!problemId) return
    getDoc(doc(db, 'solutions', problemId)).then(d => {
      if (d.exists()) {
        const data = d.data()
        setForm(prev => ({
          ...prev,
          status: data.status || 'in-progress',
          language: data.language || 'python',
          code: data.code || '',
          notes: data.notes || '',
          timeComplexity: data.timeComplexity || '',
          spaceComplexity: data.spaceComplexity || '',
          approach: data.approach || '',
        }))
      }
    }).catch(err => console.error('Load error:', err))
  }, [problemId])

  const save = async () => {
    setSaving(true)
    setError('')
    try {
      await setDoc(doc(db, 'solutions', problemId), {
        status: form.status,
        language: form.language,
        code: form.code,
        notes: form.notes,
        timeComplexity: form.timeComplexity,
        spaceComplexity: form.spaceComplexity,
        approach: form.approach,
        updatedAt: serverTimestamp()
      }, { merge: true })
      setSaved(true)
      setTimeout(() => setSaved(false), 2500)
    } catch (err) {
      console.error('Save error:', err)
      setError('Save failed: ' + err.message)
    } finally {
      setSaving(false)
    }
  }

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
          <button onClick={save} disabled={saving} style={{...s.saveBtn, background: saved ? '#00cc6a' : '#00ff88'}}>
            {saving ? 'saving...' : saved ? '✓ saved!' : 'save'}
          </button>
        </div>
      </header>

      {error && <div style={s.errorBar}>{error}</div>}

      <div style={s.container}>
        {/* Meta row */}
        <div style={s.metaRow}>
          <div style={s.field}>
            <label style={s.label}>status</label>
            <select style={s.select} value={form.status} onChange={e => setForm(f => ({...f, status: e.target.value}))}>
              <option value="unsolved">unsolved</option>
              <option value="in-progress">in progress</option>
              <option value="solved">solved</option>
            </select>
          </div>
          <div style={s.field}>
            <label style={s.label}>language</label>
            <select style={s.select} value={form.language} onChange={e => setForm(f => ({...f, language: e.target.value}))}>
              {LANGS.map(l => <option key={l} value={l}>{l}</option>)}
            </select>
          </div>
          <div style={s.field}>
            <label style={s.label}>time complexity</label>
            <input style={s.input} placeholder="O(n)" value={form.timeComplexity}
              onChange={e => setForm(f => ({...f, timeComplexity: e.target.value}))} />
          </div>
          <div style={s.field}>
            <label style={s.label}>space complexity</label>
            <input style={s.input} placeholder="O(n)" value={form.spaceComplexity}
              onChange={e => setForm(f => ({...f, spaceComplexity: e.target.value}))} />
          </div>
        </div>

        {/* Two columns */}
        <div style={s.cols}>
          {/* Code */}
          <div style={s.col}>
            <label style={s.label}>solution code</label>
            <textarea
              style={s.codeArea}
              value={form.code}
              onChange={e => setForm(f => ({...f, code: e.target.value}))}
              placeholder={`# paste your ${form.language} solution here\n\nclass Solution:\n    def twoSum(self, nums, target):\n        ...`}
              spellCheck={false}
            />
          </div>

          {/* Notes */}
          <div style={s.col}>
            <label style={s.label}>approach / explanation</label>
            <textarea
              style={{...s.codeArea, fontFamily:"'Syne',sans-serif", fontSize:'0.88rem', lineHeight:1.7}}
              value={form.approach}
              onChange={e => setForm(f => ({...f, approach: e.target.value}))}
              placeholder="Explain your approach. What pattern? Why does it work?"
            />
            <label style={{...s.label, marginTop:'1rem'}}>notes / learnings</label>
            <textarea
              style={{...s.codeArea, height:'150px', fontFamily:"'Syne',sans-serif", fontSize:'0.88rem', lineHeight:1.7}}
              value={form.notes}
              onChange={e => setForm(f => ({...f, notes: e.target.value}))}
              placeholder="What did you struggle with? Key insight? What to remember?"
            />
          </div>
        </div>

        <div style={s.saveRow}>
          <button onClick={save} disabled={saving} style={{...s.saveBtnBig, background: saved ? '#00cc6a' : '#00ff88'}}>
            {saving ? 'saving...' : saved ? '✓ saved!' : 'save solution'}
          </button>
          {error && <span style={{fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#ff4444'}}>{error}</span>}
        </div>
      </div>
    </div>
  )
}

const s = {
  page: { minHeight:'100vh', background:'#0a0a0a', color:'#f0f0f0' },
  header: { display:'flex', alignItems:'center', justifyContent:'space-between', padding:'0.9rem 2rem', borderBottom:'1px solid #1a1a1a', position:'sticky', top:0, background:'#0a0a0a', zIndex:10, flexWrap:'wrap', gap:'0.5rem' },
  back: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem', color:'#555', textDecoration:'none' },
  headerMid: { display:'flex', alignItems:'center', gap:'0.75rem', flex:1, justifyContent:'center', flexWrap:'wrap' },
  diff: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', fontWeight:700 },
  title: { fontFamily:"'Syne',sans-serif", fontSize:'1rem', fontWeight:700 },
  topic: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555' },
  headerRight: { display:'flex', gap:'0.75rem', alignItems:'center' },
  lcLink: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.7rem', color:'#555', textDecoration:'none' },
  saveBtn: { color:'#0a0a0a', border:'none', borderRadius:'6px', padding:'0.45rem 1.2rem', fontWeight:700, fontFamily:"'Syne',sans-serif", fontSize:'0.8rem', cursor:'pointer', transition:'background 0.3s' },
  errorBar: { background:'#ff444420', border:'1px solid #ff444440', color:'#ff4444', padding:'0.5rem 2rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.75rem' },
  container: { maxWidth:'1200px', margin:'0 auto', padding:'1.5rem 2rem' },
  metaRow: { display:'flex', gap:'1rem', flexWrap:'wrap', marginBottom:'1.5rem', background:'#111', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem 1.2rem' },
  field: { display:'flex', flexDirection:'column', gap:'0.3rem' },
  label: { fontFamily:"'JetBrains Mono',monospace", fontSize:'0.65rem', color:'#555', textTransform:'uppercase', letterSpacing:'0.08em' },
  select: { background:'#0a0a0a', border:'1px solid #2a2a2a', borderRadius:'5px', color:'#f0f0f0', padding:'0.4rem 0.6rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.8rem' },
  input: { background:'#0a0a0a', border:'1px solid #2a2a2a', borderRadius:'5px', color:'#f0f0f0', padding:'0.4rem 0.6rem', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.8rem', outline:'none', width:'130px' },
  cols: { display:'grid', gridTemplateColumns:'1fr 1fr', gap:'1.5rem', marginBottom:'1.5rem' },
  col: { display:'flex', flexDirection:'column', gap:'0.5rem' },
  codeArea: { background:'#0d0d0d', border:'1px solid #1a1a1a', borderRadius:'8px', padding:'1rem', color:'#f0f0f0', fontFamily:"'JetBrains Mono',monospace", fontSize:'0.82rem', lineHeight:1.7, resize:'vertical', minHeight:'340px', outline:'none' },
  saveRow: { display:'flex', alignItems:'center', gap:'1rem' },
  saveBtnBig: { color:'#0a0a0a', border:'none', borderRadius:'8px', padding:'0.7rem 2.5rem', fontWeight:700, fontFamily:"'Syne',sans-serif", fontSize:'0.95rem', cursor:'pointer', transition:'background 0.3s' },
}
