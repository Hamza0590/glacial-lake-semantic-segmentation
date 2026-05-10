import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, Upload, CheckCircle2, Cpu } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const CNN_BLOCKS = [
  { id: 'enc1',  label: 'Encoder 1',   desc: '32×200×200',  color: '#b60058', size: 160, width: 80 },
  { id: 'enc2',  label: 'Encoder 2',   desc: '64×100×100',  color: '#b60058', size: 140, width: 100 },
  { id: 'enc3',  label: 'Encoder 3',   desc: '128×50×50',   color: '#b60058', size: 120, width: 120 },
  { id: 'dec1',  label: 'Decoder 1',   desc: '64×100×100',  color: '#006970', size: 140, width: 100 },
  { id: 'dec2',  label: 'Decoder 2',   desc: '32×200×200',  color: '#006970', size: 160, width: 80 },
  { id: 'out',   label: 'Output Head', desc: '1×400×400',   color: '#2D3748', size: 180, width: 50 },
];

const ASPP_BRANCHES = [
  { id: 'b1', label: '1x1 Conv', color: '#e53e3e' }, // Red
  { id: 'b2', label: '3x3 Conv rate=6', color: '#ed64a6' }, // Pink
  { id: 'b3', label: '3x3 Conv rate=12', color: '#805ad5' }, // Purple
  { id: 'b4', label: '3x3 Conv rate=18', color: '#319795' }, // Teal
  { id: 'b5', label: 'Image Pooling', color: '#4fd1c5' }, // Light Cyan
];

const UNET_LEVELS = [
  { level: 1, enc: 'enc1', dec: 'dec1', enc_label: 'Conv 64', dec_label: 'Up-Conv 64', size: 160, width: 60, color: '#8b5cf6', encStage: 0, decStage: 8 },
  { level: 2, enc: 'enc2', dec: 'dec2', enc_label: 'Conv 128', dec_label: 'Up-Conv 128', size: 130, width: 80, color: '#6366f1', encStage: 1, decStage: 7 },
  { level: 3, enc: 'enc3', dec: 'dec3', enc_label: 'Conv 256', dec_label: 'Up-Conv 256', size: 100, width: 100, color: '#3b82f6', encStage: 2, decStage: 6 },
  { level: 4, enc: 'enc4', dec: 'dec4', enc_label: 'Conv 512', dec_label: 'Up-Conv 512', size: 70, width: 120, color: '#0ea5e9', encStage: 3, decStage: 5 },
];
const UNET_BOTTLENECK = { id: 'bottleneck', label: 'Bottleneck 1024', size: 60, width: 160, color: '#06b6d4', stage: 4 };

export default function VisualizationLab() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelType, setModelType] = useState('simple_cnn'); // 'simple_cnn' or 'aspp_segnet'
  
  // Animation state
  const [activeStage, setActiveStage] = useState(-1);

  const onFileChange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setActiveStage(-1);
  };

  const runVisualization = async () => {
    if (!file) return;
    setIsProcessing(true);
    setResult(null);
    setActiveStage(0);

    const fd = new FormData();
    fd.append('image', file);
    fd.append('model_name', modelType);
    fd.append('extract_features', 'true');

    let fetchSuccess = false;
    let data = null;

    fetch('/predict', { method: 'POST', body: fd })
      .then(res => res.json())
      .then(d => { data = d; fetchSuccess = true; })
      .catch(e => console.error("Inference Error:", e));

    const totalStages = modelType === 'simple_cnn' ? CNN_BLOCKS.length : (modelType === 'unet' ? 9 : 4); // ASPP has 4 main visual stages

    for (let i = 0; i < totalStages; i++) {
      setActiveStage(i);
      await new Promise(r => setTimeout(r, 600));
    }

    if (!fetchSuccess) await new Promise(r => setTimeout(r, 1500));
    
    setResult(data);
    setIsProcessing(false);
    setActiveStage(99); // complete
  };

  const renderSimpleCNN = () => (
    <div style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', gap: '16px', width: '100%', maxWidth: '1200px', justifyContent: 'center' }}>
      {CNN_BLOCKS.map((layer, idx) => {
        const isActive = activeStage === idx;
        const isPassed = activeStage > idx;
        return (
          <React.Fragment key={layer.id}>
            {idx > 0 && (
              <div style={{ height: '2px', flex: 1, background: '#CBD5E0', position: 'relative', minWidth: '40px' }}>
                <AnimatePresence>
                  {isActive && (
                    <motion.div
                      initial={{ left: 0, width: '0%' }} animate={{ left: '100%', width: '100%' }} transition={{ duration: 0.4, ease: "linear" }}
                      style={{ position: 'absolute', top: '-1px', height: '4px', background: '#00dcff', boxShadow: '0 0 10px #00dcff' }}
                    />
                  )}
                </AnimatePresence>
              </div>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', position: 'relative' }}>
              <motion.div
                animate={{ scale: isActive ? 1.15 : 1, boxShadow: isActive ? `0 0 40px ${layer.color}` : '0 10px 20px rgba(0,0,0,0.3)', borderColor: isActive || isPassed ? layer.color : '#000000' }}
                style={{ height: `${layer.size}px`, width: `${layer.width}px`, background: isPassed ? `${layer.color}40` : '#161e2e', border: `3px solid`, borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', transition: 'all 0.3s ease' }}
              >
                {isActive && <motion.div initial={{ opacity: 0 }} animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity, duration: 0.8 }} style={{ position: 'absolute', inset: 0, background: layer.color }} />}
                {result && result.feature_maps && result.feature_maps[layer.id] && (
                  <motion.img initial={{ opacity: 0 }} animate={{ opacity: 1 }} src={`data:image/png;base64,${result.feature_maps[layer.id]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen' }} />
                )}
              </motion.div>
              <div style={{ textAlign: 'center' }}>
                <div className="mono" style={{ fontSize: '10px', color: 'white', textTransform: 'uppercase', letterSpacing: '0.1em', fontWeight: 600 }}>{layer.label}</div>
                <div className="mono" style={{ fontSize: '8px', color: 'rgba(255,255,255,0.5)' }}>{layer.desc}</div>
              </div>
            </div>
          </React.Fragment>
        );
      })}
    </div>
  );

  const renderASPP = () => {
    const stage1 = activeStage >= 0; // Base CNN
    const stage2 = activeStage >= 1; // Branches
    const stage3 = activeStage >= 2; // Concat
    const stage4 = activeStage >= 3; // Output

    return (
      <div style={{ position: 'relative', zIndex: 10, display: 'flex', alignItems: 'center', gap: '32px', width: '100%', maxWidth: '1000px', justifyContent: 'center' }}>
        
        {/* Base Extractor */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <motion.div animate={{ borderColor: stage1 ? '#a0aec0' : '#000000', background: stage1 ? 'rgba(160,174,192,0.2)' : '#161e2e' }} style={{ height: '220px', width: '80px', border: '3px solid', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', overflow: 'hidden', borderRadius: '4px', boxShadow: '0 10px 20px rgba(0,0,0,0.3)' }}>
             {activeStage === 0 && <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity }} style={{ position: 'absolute', inset: 0, background: '#a0aec0' }} />}
             {result && result.feature_maps && result.feature_maps['backbone'] && (
               <img src={`data:image/png;base64,${result.feature_maps['backbone']}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen' }} />
             )}
          </motion.div>
          <div className="mono" style={{ fontSize: '10px', color: 'white', textTransform: 'uppercase', fontWeight: 600 }}>CNN Backbone</div>
        </div>

        {/* Splitter Line */}
        <div style={{ width: '60px', height: '400px', borderTop: '2px solid rgba(255,255,255,0.1)', borderBottom: '2px solid rgba(255,255,255,0.1)', borderLeft: '2px solid rgba(255,255,255,0.1)', position: 'relative' }}>
            <div style={{ position: 'absolute', top: '50%', left: 0, width: '100%', height: '2px', background: 'rgba(255,255,255,0.1)' }} />
            {activeStage === 1 && <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} style={{ position: 'absolute', top: '50%', left: 0, height: '2px', background: '#00dcff', boxShadow: '0 0 10px #00dcff' }} />}
        </div>

        {/* Parallel Branches */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px', height: '600px', justifyContent: 'space-between' }}>
          {ASPP_BRANCHES.map((b, i) => (
            <div key={b.id} style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              <motion.div animate={{ borderColor: stage2 ? b.color : '#000000', scale: activeStage === 1 ? 1.05 : 1 }} style={{ padding: '16px 32px', border: '3px solid', background: stage2 ? `${b.color}20` : '#161e2e', width: '260px', textAlign: 'center', position: 'relative', overflow: 'hidden', borderRadius: '4px', boxShadow: '0 5px 15px rgba(0,0,0,0.2)' }}>
                {activeStage === 1 && <motion.div animate={{ opacity: [0.2, 0.6, 0.2] }} transition={{ repeat: Infinity }} style={{ position: 'absolute', inset: 0, background: b.color }} />}
                <span className="mono" style={{ fontSize: '13px', color: 'white', position: 'relative', zIndex: 2, fontWeight: 700 }}>{b.label}</span>
              </motion.div>
              <motion.div animate={{ borderColor: stage2 ? b.color : '#000000' }} style={{ height: '100px', width: '40px', border: '3px solid', background: stage2 ? `${b.color}80` : '#161e2e', transform: 'skewY(-15deg)', position: 'relative', overflow: 'hidden', borderRadius: '2px' }}>
                {result && result.feature_maps && result.feature_maps[b.id] && (
                  <img src={`data:image/png;base64,${result.feature_maps[b.id]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen', transform: 'skewY(15deg) scale(1.5)' }} />
                )}
              </motion.div>
            </div>
          ))}
        </div>

        {/* Combiner Line */}
        <div style={{ width: '60px', height: '400px', borderTop: '2px solid rgba(255,255,255,0.1)', borderBottom: '2px solid rgba(255,255,255,0.1)', borderRight: '2px solid rgba(255,255,255,0.1)', position: 'relative' }}>
            <div style={{ position: 'absolute', top: '50%', left: 0, width: '100%', height: '2px', background: 'rgba(255,255,255,0.1)' }} />
            {activeStage === 2 && <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} style={{ position: 'absolute', top: '50%', left: 0, height: '2px', background: '#00dcff', boxShadow: '0 0 10px #00dcff' }} />}
        </div>

        {/* Concatenated Block */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px' }}>
          <div style={{ display: 'flex', transform: 'skewY(-15deg)', position: 'relative', overflow: 'hidden', borderRadius: '2px' }}>
            {ASPP_BRANCHES.map((b) => (
              <motion.div key={b.id} animate={{ borderColor: stage3 ? b.color : '#000000', background: stage3 ? `${b.color}80` : '#161e2e' }} style={{ height: '140px', width: '35px', border: '3px solid' }} />
            ))}
            {result && result.feature_maps && result.feature_maps['concat'] && (
              <img src={`data:image/png;base64,${result.feature_maps['concat']}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen', transform: 'skewY(15deg) scale(1.5)' }} />
            )}
          </div>
          <div className="mono" style={{ fontSize: '10px', color: 'white', textTransform: 'uppercase', fontWeight: 600 }}>Concat</div>
        </div>

      </div>
    );
  };

  const renderUNet = () => {
    return (
      <div style={{ position: 'relative', zIndex: 10, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', width: '100%', maxWidth: '1000px', margin: '0 auto' }}>
        {UNET_LEVELS.map((lvl) => {
          const encActive = activeStage === lvl.encStage;
          const encPassed = activeStage > lvl.encStage;
          const decActive = activeStage === lvl.decStage;
          const decPassed = activeStage > lvl.decStage;
          const skipActive = activeStage >= lvl.decStage;

          return (
            <div key={lvl.level} style={{ display: 'flex', width: '100%', justifyContent: 'space-between', alignItems: 'center', padding: `0 ${lvl.level * 40}px` }}>
              
              {/* Encoder Block */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                <motion.div animate={{ scale: encActive ? 1.1 : 1, boxShadow: encActive ? `0 0 30px ${lvl.color}` : '0 10px 20px rgba(0,0,0,0.3)', borderColor: encActive || encPassed ? lvl.color : '#000000' }}
                  style={{ height: `${lvl.size}px`, width: `${lvl.width}px`, background: encPassed ? `${lvl.color}40` : '#161e2e', border: '3px solid', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                  {encActive && <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity }} style={{ position: 'absolute', inset: 0, background: lvl.color }} />}
                  {result && result.feature_maps && result.feature_maps[lvl.enc] && (
                    <img src={`data:image/png;base64,${result.feature_maps[lvl.enc]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen' }} />
                  )}
                </motion.div>
                <div className="mono" style={{ fontSize: '10px', color: 'white', fontWeight: 600 }}>{lvl.enc_label}</div>
              </div>

              {/* Skip Connection */}
              <div style={{ flex: 1, height: '2px', position: 'relative', margin: '0 24px', display: 'flex', alignItems: 'center' }}>
                <div style={{ width: '100%', height: '2px', background: 'rgba(255,255,255,0.1)', position: 'absolute' }} />
                {skipActive && (
                  <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 0.5 }} style={{ position: 'absolute', height: '2px', background: lvl.color, boxShadow: `0 0 10px ${lvl.color}` }} />
                )}
                {/* Arrow */}
                <span className="mono" style={{ position: 'absolute', top: '-16px', left: '50%', transform: 'translateX(-50%)', fontSize: '9px', color: skipActive ? 'white' : 'rgba(255,255,255,0.3)' }}>Copy & Crop</span>
              </div>

              {/* Decoder Block */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                <motion.div animate={{ scale: decActive ? 1.1 : 1, boxShadow: decActive ? `0 0 30px ${lvl.color}` : '0 10px 20px rgba(0,0,0,0.3)', borderColor: decActive || decPassed ? lvl.color : '#000000' }}
                  style={{ height: `${lvl.size}px`, width: `${lvl.width}px`, background: decPassed ? `${lvl.color}40` : '#161e2e', border: '3px solid', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
                  {decActive && <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity }} style={{ position: 'absolute', inset: 0, background: lvl.color }} />}
                  {result && result.feature_maps && result.feature_maps[lvl.dec] && (
                    <img src={`data:image/png;base64,${result.feature_maps[lvl.dec]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen' }} />
                  )}
                </motion.div>
                <div className="mono" style={{ fontSize: '10px', color: 'white', fontWeight: 600 }}>{lvl.dec_label}</div>
              </div>

            </div>
          );
        })}

        {/* Bottleneck */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px', marginTop: '16px' }}>
          <motion.div animate={{ scale: activeStage === UNET_BOTTLENECK.stage ? 1.1 : 1, boxShadow: activeStage === UNET_BOTTLENECK.stage ? `0 0 30px ${UNET_BOTTLENECK.color}` : '0 10px 20px rgba(0,0,0,0.3)', borderColor: activeStage >= UNET_BOTTLENECK.stage ? UNET_BOTTLENECK.color : '#000000' }}
            style={{ height: `${UNET_BOTTLENECK.size}px`, width: `${UNET_BOTTLENECK.width}px`, background: activeStage > UNET_BOTTLENECK.stage ? `${UNET_BOTTLENECK.color}40` : '#161e2e', border: '3px solid', borderRadius: '4px', position: 'relative', overflow: 'hidden' }}>
            {activeStage === UNET_BOTTLENECK.stage && <motion.div animate={{ opacity: [0.5, 1, 0.5] }} transition={{ repeat: Infinity }} style={{ position: 'absolute', inset: 0, background: UNET_BOTTLENECK.color }} />}
            {result && result.feature_maps && result.feature_maps[UNET_BOTTLENECK.id] && (
              <img src={`data:image/png;base64,${result.feature_maps[UNET_BOTTLENECK.id]}`} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'cover', opacity: 0.8, mixBlendMode: 'screen' }} />
            )}
          </motion.div>
          <div className="mono" style={{ fontSize: '10px', color: 'white', fontWeight: 600 }}>{UNET_BOTTLENECK.label}</div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ minHeight: '100vh', background: '#fbf9f4', padding: '48px', display: 'flex', flexDirection: 'column', gap: '32px' }}>
      
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
          <button onClick={() => navigate('/dashboard')} style={{ padding: '12px', border: '1px solid #CBD5E0', background: 'white', cursor: 'pointer', display: 'flex', alignItems: 'center' }}>
            <ArrowLeft size={20} />
          </button>
          <div>
            <div className="mono" style={{ fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.2em', color: '#b60058', marginBottom: '4px' }}>
              Architecture Visualizer
            </div>
            <h1 style={{ fontSize: '28px', color: '#2D3748', fontWeight: 600 }}>Topology Inspector</h1>
          </div>
        </div>
      </header>

      {/* Control Strip */}
      <section style={{ border: '1px solid #CBD5E0', padding: '24px', background: 'rgba(255,255,255,0.7)', display: 'flex', alignItems: 'center', gap: '24px' }}>
        <div style={{ display: 'flex', gap: '8px', borderRight: '1px solid #CBD5E0', paddingRight: '24px' }}>
          <button 
            onClick={() => { setModelType('simple_cnn'); setActiveStage(-1); setResult(null); }}
            className="mono" style={{ padding: '8px 16px', fontSize: '10px', border: '1px solid #CBD5E0', background: modelType === 'simple_cnn' ? '#b60058' : 'white', color: modelType === 'simple_cnn' ? 'white' : '#5b3f46' }}
          >
            SIMPLE CNN
          </button>
            <button 
              onClick={() => { setModelType('aspp_segnet'); setActiveStage(-1); setResult(null); }}
              className="mono" style={{ padding: '8px 16px', fontSize: '10px', border: '1px solid #CBD5E0', background: modelType === 'aspp_segnet' ? '#b60058' : 'white', color: modelType === 'aspp_segnet' ? 'white' : '#5b3f46' }}
            >
              ASPP-SEGNET
            </button>
            <button 
              onClick={() => { setModelType('unet'); setActiveStage(-1); setResult(null); }}
              className="mono" style={{ padding: '8px 16px', fontSize: '10px', border: '1px solid #CBD5E0', background: modelType === 'unet' ? '#b60058' : 'white', color: modelType === 'unet' ? 'white' : '#5b3f46' }}
            >
              U-NET
            </button>
          </div>

        <label className="btn-outline" style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer' }}>
          <input type="file" style={{ display: 'none' }} onChange={onFileChange} accept="image/*" />
          <Upload size={16} />
          {file ? file.name : 'Upload Image'}
        </label>
        
        <button className="btn-primary" onClick={runVisualization} disabled={!file || isProcessing} style={{ display: 'flex', alignItems: 'center', gap: '8px', opacity: (!file || isProcessing) ? 0.5 : 1 }}>
          <Play size={16} />
          {isProcessing ? 'Processing...' : 'Run Visualization'}
        </button>

        {result && (
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '8px', color: '#006970' }}>
            <CheckCircle2 size={16} />
            <span className="mono" style={{ fontSize: '10px', textTransform: 'uppercase' }}>Inference Complete</span>
          </div>
        )}
      </section>

      {/* Main Visualizer Area */}
      <main style={{ flex: 1, border: '1px solid #CBD5E0', background: '#0a0f1e', position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '64px 32px', overflow: 'hidden', boxShadow: 'inset 0 0 100px rgba(0,0,0,0.5)' }}>
        <div style={{ position: 'absolute', inset: 0, backgroundImage: 'linear-gradient(to right, rgba(0, 220, 255, 0.05) 1px, transparent 1px), linear-gradient(to bottom, rgba(0, 220, 255, 0.05) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

        {modelType === 'simple_cnn' && renderSimpleCNN()}
        {modelType === 'aspp_segnet' && renderASPP()}
        {modelType === 'unet' && renderUNet()}

        {/* Input/Output Previews underneath the blocks */}
        <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', maxWidth: '1000px', marginTop: '64px', zIndex: 10 }}>
          <div style={{ width: '200px' }}>
            <div className="mono" style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px', textTransform: 'uppercase', fontWeight: 600 }}>Source Tensor</div>
            <div style={{ width: '100%', aspectRatio: '1', border: '1px solid #334', background: '#161e2e', display: 'flex', alignItems: 'center', justifyContent: 'center', borderRadius: '4px', overflow: 'hidden' }}>
              {preview ? <img src={preview} alt="input" style={{ width: '100%', height: '100%', objectFit: 'cover' }} /> : <span className="mono" style={{ fontSize: '9px', color: '#334' }}>NO DATA</span>}
            </div>
          </div>
          <div style={{ width: '200px' }}>
            <div className="mono" style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)', marginBottom: '8px', textTransform: 'uppercase', fontWeight: 600 }}>Output Mask</div>
            <div style={{ width: '100%', aspectRatio: '1', border: `1px solid ${result ? '#00dcff' : '#334'}`, background: '#161e2e', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative', borderRadius: '4px', overflow: 'hidden' }}>
              <AnimatePresence>
                {result && result.mask_image_base64 && (
                  <motion.img initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} src={`data:image/png;base64,${result.mask_image_base64}`} alt="output" style={{ width: '100%', height: '100%', objectFit: 'cover', position: 'absolute', inset: 0 }} />
                )}
              </AnimatePresence>
              {!result && <span className="mono" style={{ fontSize: '9px', color: '#334' }}>WAITING...</span>}
            </div>
          </div>
        </div>

      </main>
    </div>
  );
}
