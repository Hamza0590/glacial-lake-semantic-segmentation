import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, BarChart2, Zap, Target, Download, Upload, Play, CheckCircle2 } from 'lucide-react';
import { motion } from 'framer-motion';

const VIEW_MODES = [
  { id: 'mask',    label: 'Binary Mask' },
  { id: 'colored', label: 'Lake Map'    },
  { id: 'overlay', label: 'Overlay'     },
];

const modelMetadata = {
  'unet':        { display: 'U-Net',       description: 'Encoder-decoder with skip connections.', speed: '12ms' },
  'simple_cnn':  { display: 'Simple CNN',  description: 'Lightweight baseline, fast inference.',   speed: '8ms'  },
  'aspp_segnet': { display: 'ASPP-SegNet', description: 'Multi-scale atrous pyramid pooling.',     speed: '42ms' },
};

const MODEL_IDS = ['unet', 'simple_cnn', 'aspp_segnet'];

// ── Inline style helpers ──────────────────────────────────────────────────────
const styles = {
  page: {
    minHeight: '100vh',
    background: '#fbf9f4',
    padding: '48px',
    display: 'flex',
    flexDirection: 'column',
    gap: '48px',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: '24px' },
  backBtn: {
    padding: '12px',
    border: '1px solid #CBD5E0',
    background: 'white',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
  },
  uploadSection: {
    border: '1px solid #CBD5E0',
    padding: '32px',
    background: 'rgba(255,255,255,0.7)',
    display: 'flex',
    gap: '32px',
    alignItems: 'flex-start',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '32px',
  },
  card: {
    display: 'flex',
    flexDirection: 'column',
    background: 'rgba(255,255,255,0.85)',
    border: '1px solid #CBD5E0',
  },
  cardHeader: {
    padding: '20px 24px',
    borderBottom: '1px solid #CBD5E0',
    background: 'white',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  tabBar: {
    display: 'flex',
    borderBottom: '1px solid #CBD5E0',
    background: '#fafafa',
  },
  tabActive: {
    flex: 1,
    padding: '10px 4px',
    fontSize: '9px',
    fontFamily: 'JetBrains Mono, monospace',
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    fontWeight: 700,
    border: 'none',
    borderBottom: '2px solid #b60058',
    background: 'white',
    color: '#b60058',
    cursor: 'pointer',
  },
  tabInactive: {
    flex: 1,
    padding: '10px 4px',
    fontSize: '9px',
    fontFamily: 'JetBrains Mono, monospace',
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    fontWeight: 400,
    border: 'none',
    borderBottom: '2px solid transparent',
    background: 'transparent',
    color: '#5b3f46',
    cursor: 'pointer',
  },
  viewport: {
    position: 'relative',
    aspectRatio: '16/9',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
  },
  viewportDark: { background: '#0a0a14' },
  viewportLight: { background: '#f8f8f8' },
  badge: {
    position: 'absolute',
    top: '8px',
    right: '8px',
    padding: '2px 8px',
    fontSize: '9px',
    fontFamily: 'JetBrains Mono, monospace',
    textTransform: 'uppercase',
    color: 'white',
    letterSpacing: '0.1em',
  },
  metricsPanel: { padding: '32px', flex: 1 },
  divider: { height: '1px', background: '#e2e8f0', margin: '16px 0' },
  label: {
    fontSize: '10px',
    fontFamily: 'JetBrains Mono, monospace',
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    color: '#5b3f46',
  },
  bigNumber: {
    fontSize: '24px',
    fontFamily: 'JetBrains Mono, monospace',
    fontWeight: 700,
    color: '#2D3748',
  },
};

// ── Component ─────────────────────────────────────────────────────────────────
export default function ComparisonLab() {
  const navigate = useNavigate();
  const [modelData,       setModelData]       = React.useState([]);
  const [loading,         setLoading]         = React.useState(true);
  const [inferenceFile,   setInferenceFile]   = React.useState(null);
  const [inferencePreview,setInferencePreview]= React.useState(null);
  const [inferenceResults,setInferenceResults]= React.useState({});
  const [isInferencing,   setIsInferencing]   = React.useState(false);
  // Independent view mode per model card
  const [viewModes, setViewModes] = React.useState({
    unet: 'mask', simple_cnn: 'mask', aspp_segnet: 'mask',
  });

  React.useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      try {
        const results = await Promise.all(
          MODEL_IDS.map(async (id) => {
            const res = await fetch(`/evaluate/${id}`);
            if (!res.ok) return { id, error: true };
            return { id, ...(await res.json()) };
          })
        );
        setModelData(results);
      } catch (e) { console.error(e); }
      finally { setLoading(false); }
    };
    fetchAll();
  }, []);

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setInferenceFile(file);
    setInferencePreview(URL.createObjectURL(file));
    setInferenceResults({});
  };

  const runMultiInference = async () => {
    if (!inferenceFile) return;
    setIsInferencing(true);
    try {
      const settled = await Promise.all(
        MODEL_IDS.map(async (id) => {
          const fd = new FormData();
          fd.append('image', inferenceFile);
          fd.append('model_name', id);
          const res = await fetch('/predict', { method: 'POST', body: fd });
          if (!res.ok) return { id, error: true };
          return res.json();
        })
      );
      const map = {};
      settled.forEach(r => { map[r.model_name || r.id] = r; });
      setInferenceResults(map);
    } catch (e) {
      console.error(e);
      alert('Inference failed — is the API running?');
    } finally {
      setIsInferencing(false);
    }
  };

  const getImg = (live, mode) => {
    if (!live || live.error) return null;
    const key = mode === 'mask' ? 'mask_image_base64' : mode === 'colored' ? 'colored_mask_base64' : 'overlay_image_base64';
    return live[key] ? `data:image/png;base64,${live[key]}` : null;
  };

  return (
    <div style={styles.page}>

      {/* ── Header ── */}
      <header style={styles.header}>
        <div style={styles.headerLeft}>
          <button style={styles.backBtn} onClick={() => navigate('/dashboard')}>
            <ArrowLeft size={20} />
          </button>
          <div>
            <div className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.2em', color: '#b60058', marginBottom: '4px' }}>
              Laboratory Environment
            </div>
            <h1 style={{ fontSize: '28px', color: '#2D3748', fontWeight: 600 }}>Multi-Model Instrument Lab</h1>
          </div>
        </div>
        <button className="btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Download size={16} /> Export Protocol
        </button>
      </header>

      {/* ── Upload Section ── */}
      <section style={styles.uploadSection}>
        <div style={{ flex: 1 }}>
          <div className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.2em', color: '#b60058', fontWeight: 700, marginBottom: '12px' }}>
            Simultaneous Inference Test
          </div>
          <p style={{ fontSize: '13px', color: '#5b3f46', maxWidth: '520px', marginBottom: '20px' }}>
            Upload a Sentinel-2 satellite patch to run inference across all three models in parallel.
            Use the <strong>Binary Mask / Lake Map / Overlay</strong> tabs on each card to switch views.
          </p>
          <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
            <label className="btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
              <input type="file" style={{ display: 'none' }} onChange={onFileChange} accept="image/*" />
              <Upload size={16} />
              {inferenceFile ? inferenceFile.name : 'Select Image'}
            </label>
            <button
              className="btn-primary"
              onClick={runMultiInference}
              disabled={!inferenceFile || isInferencing}
              style={{ display: 'flex', alignItems: 'center', gap: '8px', opacity: (!inferenceFile || isInferencing) ? 0.5 : 1 }}
            >
              {isInferencing ? <Zap size={16} /> : <Play size={16} />}
              {isInferencing ? 'Running…' : 'Run All Models'}
            </button>
          </div>
        </div>

        {inferencePreview && (
          <div style={{ width: '160px', height: '160px', border: '1px solid #CBD5E0', overflow: 'hidden', flexShrink: 0 }}>
            <img src={inferencePreview} style={{ width: '100%', height: '100%', objectFit: 'cover' }} alt="preview" />
          </div>
        )}
      </section>

      {/* ── Model Cards Grid ── */}
      <div style={styles.grid}>
        {MODEL_IDS.map((id, idx) => {
          const meta     = modelMetadata[id];
          const baseline = modelData.find(d => d.id === id);
          const live     = inferenceResults[id];
          const mode     = viewModes[id];
          const displayImg = getImg(live, mode);
          const hasResult  = live && !live.error;

          return (
            <motion.div
              key={id}
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              style={styles.card}
            >
              {/* Card Header */}
              <div style={styles.cardHeader}>
                <span className="mono" style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.15em', fontWeight: 700 }}>
                  {meta.display}
                </span>
                {hasResult && <CheckCircle2 size={16} color="#006970" />}
              </div>

              {/* ── VIEW TOGGLE TABS ── always rendered, greyed when no result ── */}
              <div style={styles.tabBar}>
                {VIEW_MODES.map(vm => (
                  <button
                    key={vm.id}
                    onClick={() => setViewModes(prev => ({ ...prev, [id]: vm.id }))}
                    disabled={!hasResult}
                    style={mode === vm.id && hasResult ? styles.tabActive : { ...styles.tabInactive, opacity: hasResult ? 1 : 0.35 }}
                  >
                    {vm.label}
                  </button>
                ))}
              </div>

              {/* Viewport */}
              <div style={{ ...styles.viewport, ...(mode !== 'overlay' ? styles.viewportDark : styles.viewportLight) }}>
                {displayImg ? (
                  <>
                    <img
                      src={displayImg}
                      alt={mode}
                      style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block' }}
                    />
                    <div style={{ ...styles.badge, background: mode === 'mask' ? '#475569' : mode === 'colored' ? '#0891b2' : '#dc2626' }}>
                      {vm => vm}
                      {VIEW_MODES.find(v => v.id === mode)?.label}
                    </div>
                  </>
                ) : baseline?.predictions_image_path ? (
                  <>
                    <img
                      src={`/${baseline.predictions_image_path}`}
                      alt="baseline"
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                    />
                    <div style={{ ...styles.badge, background: '#475569' }}>Baseline Grid</div>
                  </>
                ) : (
                  <div className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', opacity: 0.3, textAlign: 'center', padding: '16px', color: '#5b3f46' }}>
                    {baseline?.error ? 'No Checkpoint — Train First' : hasResult && live.error ? 'Inference Error' : 'Upload image & run inference'}
                  </div>
                )}
              </div>

              {/* Metrics Panel */}
              <div style={styles.metricsPanel}>
                <div className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#b60058', fontWeight: 700, opacity: 0.7, marginBottom: '16px' }}>
                  Live Analytics
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '16px' }}>
                  <div>
                    <div style={styles.label}>Lake Coverage</div>
                    <div style={styles.bigNumber}>{hasResult ? `${live.lake_coverage_percent}%` : '---'}</div>
                  </div>
                  <div>
                    <div style={styles.label}>Speed</div>
                    <div style={styles.bigNumber}>{hasResult ? meta.speed : '---'}</div>
                  </div>
                </div>

                <div style={styles.divider} />

                <div className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.15em', opacity: 0.5, marginBottom: '12px' }}>
                  Checkpoint Baseline
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', opacity: 0.65 }}>
                  <div>
                    <div style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <Target size={11} /> Val IoU
                    </div>
                    <div className="mono" style={{ fontSize: '18px', fontWeight: 700 }}>{baseline?.val_iou ?? '---'}</div>
                  </div>
                  <div>
                    <div style={{ ...styles.label, display: 'flex', alignItems: 'center', gap: '4px' }}>
                      <BarChart2 size={11} /> Val F1
                    </div>
                    <div className="mono" style={{ fontSize: '18px', fontWeight: 700 }}>{baseline?.val_f1 ?? '---'}</div>
                  </div>
                </div>

                <div style={{ marginTop: '24px', paddingTop: '16px', borderTop: '1px solid #e2e8f0' }}>
                  <p style={{ fontSize: '11px', color: '#5b3f46', fontStyle: 'italic', opacity: 0.6, lineHeight: 1.6 }}>
                    {meta.description}
                  </p>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* ── Mask Legend ── */}
      {Object.keys(inferenceResults).length > 0 && (
        <div style={{ border: '1px solid #CBD5E0', padding: '20px 32px', background: 'rgba(255,255,255,0.5)', display: 'flex', alignItems: 'center', gap: '48px', flexWrap: 'wrap' }}>
          <span className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#b60058', fontWeight: 700 }}>Mask Legend</span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}>
            <span style={{ display: 'inline-block', width: '28px', height: '14px', background: 'white', border: '1px solid #ccc' }} />
            Lake pixel (value = 1)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}>
            <span style={{ display: 'inline-block', width: '28px', height: '14px', background: '#0a0a14', border: '1px solid #334' }} />
            Background (value = 0)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '11px', fontFamily: 'JetBrains Mono, monospace' }}>
            <span style={{ display: 'inline-block', width: '28px', height: '14px', background: '#00dcff' }} />
            Lake Map view (cyan = lake)
          </span>
          <span className="mono" style={{ marginLeft: 'auto', fontSize: '9px', color: '#5b3f46', opacity: 0.6 }}>
            Threshold: 0.5 · Input: 400×400px
          </span>
        </div>
      )}

      {/* ── Performance Summary ── */}
      <div style={{ border: '1px solid #CBD5E0', padding: '32px', background: 'rgba(255,255,255,0.4)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '32px' }}>
          <h3 className="mono" style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.2em', color: '#b60058', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <BarChart2 size={16} /> System Performance Summary
          </h3>
          <span className="mono" style={{ fontSize: '9px', color: '#5b3f46', background: 'rgba(255,255,255,0.6)', padding: '4px 12px', border: '1px solid #CBD5E0' }}>
            {loading ? 'Loading…' : `${modelData.filter(d => !d.error).length} / 3 Models Ready`}
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '32px' }}>
          {[
            { label: 'Top Precision',  value: modelData.filter(d => !d.error).sort((a,b) => b.val_iou - a.val_iou)[0]?.id?.replace('_',' ') || '---', delta: 'Highest IoU' },
            { label: 'Top Confidence', value: modelData.filter(d => !d.error).sort((a,b) => b.val_f1 - a.val_f1)[0]?.id?.replace('_',' ')  || '---', delta: 'Highest F1' },
            { label: 'Most Efficient', value: 'Simple CNN', delta: '8ms Latency' },
            { label: 'Architecture',   value: 'Hybrid',     delta: 'Parallel Execution' },
          ].map(s => (
            <div key={s.label}>
              <div className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', opacity: 0.5, marginBottom: '6px' }}>{s.label}</div>
              <div className="mono" style={{ fontSize: '18px', fontWeight: 700, textTransform: 'uppercase' }}>{s.value}</div>
              <div className="mono" style={{ fontSize: '10px', color: '#006970', fontWeight: 700 }}>{s.delta}</div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
}
