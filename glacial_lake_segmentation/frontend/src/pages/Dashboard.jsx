import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, Layers, Map as MapIcon, Settings, Download,
  Upload, Cpu, RefreshCcw, Home
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const VIEW_MODES = [
  { id: 'mask',    label: 'Binary Mask' },
  { id: 'colored', label: 'Lake Map'    },
  { id: 'overlay', label: 'Overlay'     },
];

const MODEL_MAP = {
  'U-Net':       'unet',
  'Simple CNN':  'simple_cnn',
  'ASPP-SegNet': 'aspp_segnet',
};

export default function Dashboard() {
  const navigate = useNavigate();
  const [selectedModel, setSelectedModel] = useState('unet');
  const [isProcessing,  setIsProcessing]  = useState(false);
  const [selectedFile,  setSelectedFile]  = useState(null);
  const [previewUrl,    setPreviewUrl]    = useState(null);
  const [result,        setResult]        = useState(null);
  const [viewMode,      setViewMode]      = useState('mask');
  const [logs,          setLogs]          = useState([
    '[04:12:01] SEED INITIALIZED (42)',
    '[04:12:02] READY FOR INFERENCE.',
  ]);

  const addLog = (msg) => {
    const t = new Date().toLocaleTimeString([], { hour12: false });
    setLogs(prev => [...prev.slice(-4), `[${t}] ${msg}`]);
  };

  const onFileChange = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    addLog(`LOADED: ${file.name.toUpperCase()}`);
  };

  const runInference = async () => {
    if (!selectedFile) { alert('Please upload an image first.'); return; }
    setIsProcessing(true);
    setResult(null);
    addLog(`PROCESSING: ${selectedModel.toUpperCase()}`);
    const fd = new FormData();
    fd.append('image', selectedFile);
    fd.append('model_name', selectedModel);
    try {
      const res = await fetch('/predict', { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Inference failed'); }
      const data = await res.json();
      setResult(data);
      setViewMode('mask'); // reset to binary mask view on new result
      addLog(`SUCCESS: ${data.model_name.toUpperCase()} — ${data.lake_coverage_percent}% lake`);
    } catch (e) {
      console.error(e);
      addLog(`ERROR: ${e.message.toUpperCase()}`);
      alert(e.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const getDisplaySrc = () => {
    if (!result) return null;
    if (viewMode === 'mask')    return result.mask_image_base64    ? `data:image/png;base64,${result.mask_image_base64}`    : null;
    if (viewMode === 'colored') return result.colored_mask_base64  ? `data:image/png;base64,${result.colored_mask_base64}`  : null;
    if (viewMode === 'overlay') return result.overlay_image_base64 ? `data:image/png;base64,${result.overlay_image_base64}` : null;
    return null;
  };

  const displaySrc = getDisplaySrc();

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden', background: '#fbf9f4' }}>

      {/* ── Icon Sidebar ── */}
      <aside style={{ width: '64px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '24px 0', gap: '32px', borderRight: '1px solid #CBD5E0', background: 'white', zIndex: 20 }}>
        <div style={{ cursor: 'pointer', color: '#b60058' }} onClick={() => navigate('/')}>
          <Home size={24} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          <div style={{ color: '#b60058' }}><MapIcon size={24} /></div>
          <div style={{ color: '#5b3f46', cursor: 'pointer' }}><Layers size={24} /></div>
          <div style={{ color: '#5b3f46', cursor: 'pointer' }} onClick={() => navigate('/comparison')}><Activity size={24} /></div>
        </div>
        <div style={{ marginTop: 'auto' }}>
          <div style={{ color: '#5b3f46', cursor: 'pointer' }}><Settings size={24} /></div>
        </div>
      </aside>

      {/* ── Left Control Panel ── */}
      <aside style={{ width: '320px', height: '100%', display: 'flex', flexDirection: 'column', borderRight: '1px solid #CBD5E0', background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(8px)', zIndex: 10 }}>
        <div style={{ padding: '24px', borderBottom: '1px solid #CBD5E0' }}>
          <h2 className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#5b3f46' }}>Data Input</h2>
        </div>

        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Upload Zone */}
          <label style={{ height: '180px', border: '1px dashed #CBD5E0', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '8px', cursor: 'pointer', background: 'rgba(0,0,0,0.02)', transition: 'background 0.2s' }}>
            <input type="file" style={{ display: 'none' }} onChange={onFileChange} accept="image/*" />
            <Upload size={32} color="#5b3f46" />
            <div className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', textAlign: 'center', letterSpacing: '0.08em' }}>
              {selectedFile ? selectedFile.name : 'Drop Sentinel-2 Patch'}
            </div>
            <div className="mono" style={{ fontSize: '8px', color: '#5b3f46', opacity: 0.5 }}>400×400px recommended</div>
          </label>

          {/* Model Selection */}
          <div>
            <label className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#5b3f46', display: 'block', marginBottom: '12px' }}>Processor</label>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {Object.entries(MODEL_MAP).map(([display, key]) => (
                <div
                  key={key}
                  onClick={() => setSelectedModel(key)}
                  style={{
                    padding: '12px 16px',
                    border: `1px solid ${selectedModel === key ? '#b60058' : '#CBD5E0'}`,
                    background: selectedModel === key ? 'rgba(182,0,88,0.05)' : 'white',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.15s',
                  }}
                >
                  <Cpu size={16} color={selectedModel === key ? '#b60058' : '#5b3f46'} />
                  <span className="mono" style={{ fontSize: '11px', textTransform: 'uppercase' }}>{display}</span>
                  {selectedModel === key && <div style={{ width: '8px', height: '8px', background: '#b60058', marginLeft: 'auto' }} />}
                </div>
              ))}
            </div>
          </div>

          <button
            onClick={runInference}
            disabled={isProcessing || !selectedFile}
            className="btn-primary"
            style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', width: '100%', opacity: (isProcessing || !selectedFile) ? 0.5 : 1 }}
          >
            {isProcessing ? <RefreshCcw size={16} style={{ animation: 'spin 1s linear infinite' }} /> : null}
            {isProcessing ? 'Running…' : 'Run Segmentation'}
          </button>
        </div>

        {/* Hardware bar */}
        <div style={{ marginTop: 'auto', padding: '24px', borderTop: '1px solid #CBD5E0', background: 'rgba(182,0,88,0.03)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span className="mono" style={{ fontSize: '9px', textTransform: 'uppercase' }}>Hardware Utilization</span>
            <span className="mono" style={{ fontSize: '9px' }}>{isProcessing ? '89%' : '12%'}</span>
          </div>
          <div style={{ height: '3px', background: '#e2e8f0', borderRadius: '2px' }}>
            <motion.div animate={{ width: isProcessing ? '89%' : '12%' }} style={{ height: '100%', background: '#b60058', borderRadius: '2px' }} />
          </div>
        </div>
      </aside>

      {/* ── Main Viewport ── */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', position: 'relative' }}>

        {/* ── VIEW TOGGLE BAR (always visible at top) ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '10px 24px', background: 'rgba(255,255,255,0.9)', borderBottom: '1px solid #CBD5E0', zIndex: 30 }}>
          <span className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', color: '#5b3f46', marginRight: '8px' }}>View Mode:</span>

          {VIEW_MODES.map(vm => (
            <button
              key={vm.id}
              onClick={() => setViewMode(vm.id)}
              disabled={!result}
              className="mono"
              style={{
                padding: '7px 16px',
                fontSize: '10px',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                border: '1px solid #CBD5E0',
                background: viewMode === vm.id && result ? '#b60058' : 'white',
                color:      viewMode === vm.id && result ? 'white'   : '#5b3f46',
                fontWeight: viewMode === vm.id && result ? 700       : 400,
                cursor: result ? 'pointer' : 'not-allowed',
                opacity: result ? 1 : 0.4,
                transition: 'all 0.15s',
              }}
            >
              {vm.label}
            </button>
          ))}

          <div style={{ marginLeft: 'auto' }}>
            <button
              disabled={!result}
              onClick={() => {
                const src = getDisplaySrc();
                if (!src) return;
                const a = document.createElement('a');
                a.href = src;
                a.download = `${selectedModel}_${viewMode}.png`;
                a.click();
              }}
              style={{
                display: 'flex', alignItems: 'center', gap: '8px',
                padding: '7px 16px',
                border: '1px solid #CBD5E0',
                background: 'white',
                fontSize: '10px',
                fontFamily: 'JetBrains Mono, monospace',
                textTransform: 'uppercase',
                letterSpacing: '0.1em',
                cursor: result ? 'pointer' : 'not-allowed',
                opacity: result ? 1 : 0.4,
              }}
            >
              <Download size={14} />
              Download {viewMode === 'mask' ? 'Mask' : viewMode === 'colored' ? 'Lake Map' : 'Overlay'}
            </button>
          </div>
        </div>

        {/* ── Image Frame ── */}
        <div style={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '48px',
          backgroundImage: 'linear-gradient(to right, #CBD5E0 1px, transparent 1px), linear-gradient(to bottom, #CBD5E0 1px, transparent 1px)',
          backgroundSize: '20px 20px',
          overflow: 'hidden',
          position: 'relative',
        }}>
          <div style={{
            position: 'relative',
            border: '1px solid #CBD5E0',
            background: result && viewMode !== 'overlay' ? '#0a0a14' : 'white',
            boxShadow: '0 20px 60px rgba(0,0,0,0.15)',
            maxWidth: '70%',
            maxHeight: '75vh',
            overflow: 'hidden',
            transition: 'background 0.3s',
          }}>
            <AnimatePresence mode="wait">
              {displaySrc ? (
                <motion.img
                  key={viewMode}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  src={displaySrc}
                  alt={viewMode}
                  style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh' }}
                />
              ) : previewUrl ? (
                <img
                  src={previewUrl}
                  alt="preview"
                  style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh', filter: 'grayscale(0.2)' }}
                />
              ) : (
                <div className="mono" style={{ width: '400px', height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '10px', textTransform: 'uppercase', opacity: 0.3 }}>
                  Awaiting Input Data
                </div>
              )}
            </AnimatePresence>

            {/* Scanline during inference */}
            {isProcessing && (
              <motion.div
                initial={{ left: 0 }}
                animate={{ left: '100%' }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                style={{ position: 'absolute', top: 0, width: '4px', height: '100%', background: '#ff3385', boxShadow: '0 0 20px #ff3385', zIndex: 10 }}
              />
            )}

            {/* View label badge */}
            {result && (
              <div className="mono" style={{
                position: 'absolute', top: '8px', right: '8px',
                padding: '3px 10px',
                fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.12em',
                color: 'white',
                background: viewMode === 'mask' ? '#475569' : viewMode === 'colored' ? '#0891b2' : '#dc2626',
              }}>
                {VIEW_MODES.find(v => v.id === viewMode)?.label}
              </div>
            )}
          </div>

          {/* Legend strip */}
          {result && (
            <div className="mono" style={{ position: 'absolute', bottom: '12px', left: 0, width: '100%', display: 'flex', justifyContent: 'center', gap: '24px', fontSize: '9px', textTransform: 'uppercase', color: '#5b3f46' }}>
              {viewMode === 'mask' && (
                <>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ display: 'inline-block', width: '20px', height: '10px', background: 'white', border: '1px solid #ccc' }} />Lake (1)
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ display: 'inline-block', width: '20px', height: '10px', background: '#0a0a14', border: '1px solid #334' }} />Background (0)
                  </span>
                </>
              )}
              {viewMode === 'colored' && (
                <>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ display: 'inline-block', width: '20px', height: '10px', background: '#00dcff' }} />Lake
                  </span>
                  <span style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span style={{ display: 'inline-block', width: '20px', height: '10px', background: '#0a0a14' }} />Background
                  </span>
                </>
              )}
              {viewMode === 'overlay' && <span>Cyan = lake · Red = boundary · Threshold = 0.5</span>}
              <span>Coverage: {result.lake_coverage_percent}% · 400×400px</span>
            </div>
          )}
        </div>
      </main>

      {/* ── Right Telemetry Panel ── */}
      <aside style={{ width: '300px', height: '100%', display: 'flex', flexDirection: 'column', borderLeft: '1px solid #CBD5E0', background: 'rgba(255,255,255,0.85)', backdropFilter: 'blur(8px)' }}>
        <div style={{ padding: '24px', borderBottom: '1px solid #CBD5E0' }}>
          <h2 className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.15em', color: '#5b3f46' }}>Telemetry Analysis</h2>
        </div>

        <div style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {/* Lake Coverage */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
              <span className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', color: '#5b3f46' }}>Lake Coverage</span>
              <span className="mono" style={{ fontSize: '20px', fontWeight: 700, color: '#b60058' }}>
                {result ? `${result.lake_coverage_percent}%` : '0.00%'}
              </span>
            </div>
            <div style={{ height: '4px', background: '#e2e8f0' }}>
              <motion.div
                animate={{ width: result ? `${Math.min(result.lake_coverage_percent, 100)}%` : '0%' }}
                style={{ height: '100%', background: '#b60058' }}
              />
            </div>
          </div>

          {/* Engine Status */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
              <span className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', color: '#5b3f46' }}>Engine Status</span>
              <span className="mono" style={{ fontSize: '20px', fontWeight: 700, color: '#006970' }}>
                {isProcessing ? 'RUNNING' : result ? 'DONE' : 'IDLE'}
              </span>
            </div>
            <div style={{ height: '4px', background: '#e2e8f0' }}>
              <motion.div animate={{ width: isProcessing ? '100%' : '0%' }} style={{ height: '100%', background: '#006970' }} />
            </div>
          </div>

          {/* Anomaly Detection */}
          <div style={{ border: '1px solid #CBD5E0', padding: '16px', background: 'rgba(0,0,0,0.01)' }}>
            <h4 className="mono" style={{ fontSize: '9px', textTransform: 'uppercase', letterSpacing: '0.12em', opacity: 0.5, marginBottom: '16px' }}>Anomaly Detection</h4>
            {[
              { type: 'Glacial Lake',      conf: result ? 'DETECTED' : '---', color: '#b60058' },
              { type: 'Supraglacial Pond', conf: '---',                       color: '#006970' },
              { type: 'Debris Cover',      conf: '---',                       color: '#5b3f46' },
            ].map(item => (
              <div key={item.type} style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                <div style={{ width: '8px', height: '8px', background: item.color, flexShrink: 0 }} />
                <span className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', flex: 1 }}>{item.type}</span>
                <span className="mono" style={{ fontSize: '10px' }}>{item.conf}</span>
              </div>
            ))}
          </div>

          {/* Log */}
          <div style={{ marginTop: 'auto' }}>
            <div className="mono" style={{ fontSize: '8px', textTransform: 'uppercase', opacity: 0.5, marginBottom: '8px' }}>System Log</div>
            {logs.map((log, i) => (
              <div key={i} className="mono" style={{ fontSize: '8px', color: '#5b3f46', opacity: 0.7, lineHeight: 1.8 }}>{log}</div>
            ))}
          </div>
        </div>
      </aside>

    </div>
  );
}
