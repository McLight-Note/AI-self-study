import { useState, useRef } from 'react';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { ref, uploadBytes, getDownloadURL } from 'firebase/storage';
import { db, storage } from '../firebase';

const LANGUAGES = ['Python', 'JavaScript', 'Java', 'C++', 'Go', 'Rust'];

export default function SolutionEditor({ problem, existing, onSaved }) {
  const [code, setCode] = useState(existing?.code || '');
  const [notes, setNotes] = useState(existing?.notes || '');
  const [lang, setLang] = useState(existing?.language || 'Python');
  const [status, setStatus] = useState(existing?.status || 'solved');
  const [timeComp, setTimeComp] = useState(existing?.timeComplexity || '');
  const [spaceComp, setSpaceComp] = useState(existing?.spaceComplexity || '');
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const fileRef = useRef();

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const ext = file.name.split('.').pop().toLowerCase();
    const langMap = { py: 'Python', js: 'JavaScript', java: 'Java', cpp: 'C++', go: 'Go', rs: 'Rust' };
    if (langMap[ext]) setLang(langMap[ext]);
    const text = await file.text();
    setCode(text);
  };

  const handleSave = async () => {
    if (!code.trim() && !notes.trim()) return;
    setSaving(true);
    try {
      await setDoc(doc(db, 'solutions', problem.id), {
        problemId: problem.id,
        problemTitle: problem.title,
        topic: problem.topic,
        month: problem.month,
        difficulty: problem.difficulty,
        code,
        notes,
        language: lang,
        status,
        timeComplexity: timeComp,
        spaceComplexity: spaceComp,
        updatedAt: serverTimestamp(),
        createdAt: existing?.createdAt || serverTimestamp(),
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      if (onSaved) onSaved();
    } catch (err) {
      console.error(err);
      alert('Save failed: ' + err.message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
        <select value={lang} onChange={e => setLang(e.target.value)} style={{ width: 'auto', minWidth: 120 }}>
          {LANGUAGES.map(l => <option key={l}>{l}</option>)}
        </select>
        <select value={status} onChange={e => setStatus(e.target.value)} style={{ width: 'auto', minWidth: 130 }}>
          <option value="solved">Solved</option>
          <option value="attempted">Attempted</option>
          <option value="needs-review">Needs Review</option>
        </select>
        <input
          value={timeComp}
          onChange={e => setTimeComp(e.target.value)}
          placeholder="Time: O(n)"
          style={{ width: 130, fontFamily: 'var(--font-mono)', fontSize: 13 }}
        />
        <input
          value={spaceComp}
          onChange={e => setSpaceComp(e.target.value)}
          placeholder="Space: O(1)"
          style={{ width: 130, fontFamily: 'var(--font-mono)', fontSize: 13 }}
        />
        <button
          className="btn btn-ghost"
          style={{ fontSize: 12, padding: '8px 14px' }}
          onClick={() => fileRef.current.click()}
        >
          Upload file
        </button>
        <input ref={fileRef} type="file" accept=".py,.js,.java,.cpp,.go,.rs,.txt" style={{ display: 'none' }} onChange={handleFileUpload} />
      </div>

      {/* Code editor */}
      <div style={{ position: 'relative' }}>
        <div style={{
          position: 'absolute', top: 0, left: 0, right: 0,
          background: 'var(--bg3)', borderRadius: '8px 8px 0 0',
          borderBottom: '1px solid var(--border)',
          padding: '6px 14px', display: 'flex', alignItems: 'center', gap: 8
        }}>
          <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            {lang.toLowerCase()}
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {['#ff5f57','#febc2e','#28c840'].map(c => (
              <div key={c} style={{ width: 10, height: 10, borderRadius: '50%', background: c }} />
            ))}
          </div>
        </div>
        <textarea
          value={code}
          onChange={e => setCode(e.target.value)}
          placeholder={`# Paste your ${lang} solution here...\n# Or upload a file above`}
          style={{
            minHeight: 320, paddingTop: 44,
            fontFamily: 'var(--font-mono)', fontSize: 13,
            lineHeight: 1.7, resize: 'vertical',
            borderRadius: 8, tabSize: 4,
            background: 'var(--bg3)',
          }}
          onKeyDown={e => {
            if (e.key === 'Tab') {
              e.preventDefault();
              const s = e.target.selectionStart;
              const val = code.substring(0, s) + '    ' + code.substring(s);
              setCode(val);
              setTimeout(() => e.target.selectionStart = e.target.selectionEnd = s + 4, 0);
            }
          }}
        />
      </div>

      {/* Notes */}
      <div>
        <label style={{ fontSize: 12, color: 'var(--text2)', fontWeight: 500, display: 'block', marginBottom: 6 }}>
          NOTES & EXPLANATION
        </label>
        <textarea
          value={notes}
          onChange={e => setNotes(e.target.value)}
          placeholder="Explain your approach, what tripped you up, key insight..."
          style={{ minHeight: 100, resize: 'vertical', lineHeight: 1.7 }}
        />
      </div>

      {/* Save */}
      <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn btn-primary" onClick={handleSave} disabled={saving}>
          {saving ? 'Saving...' : saved ? '✓ Saved!' : 'Save Solution'}
        </button>
      </div>
    </div>
  );
}
