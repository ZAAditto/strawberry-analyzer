import React, { useState, useRef } from 'react';
import { Upload, ChevronDown, Leaf, X, Camera, Loader2, Info, CheckCircle, AlertTriangle } from 'lucide-react';

// LSU AgCenter Logo
const LSU_LOGO = "https://cdne-agc-apps-prod.azureedge.net/assets/images/lsuagLogo/lsuagLogo-border-125.png";

// STREAMLINED Health Categories - Only Important & Detectable Features
const healthCategories = {
  colorBased: {
    title: 'COLOR-BASED INDICATORS',
    keys: ['chlorophyll', 'chlorosis', 'necrosis', 'interveinal', 'marginal']
  },
  structural: {
    title: 'LEAF STRUCTURE',
    keys: ['leafSize', 'shapeDistortion', 'edgeIrreg']
  },
  disease: {
    title: 'MAJOR DISEASES (Top 4)',
    keys: ['leafSpot', 'powderyMildew', 'anthracnose', 'angularSpot']
  },
  nutrient: {
    title: 'KEY NUTRIENTS',
    keys: ['nitrogen', 'potassium', 'iron', 'magnesium']
  },
  physical: {
    title: 'PEST DAMAGE',
    keys: ['chewing', 'stippling']
  },
  environmental: {
    title: 'ENVIRONMENTAL STRESS',
    keys: ['sunburn', 'waterStress']
  }
};

// Label mappings
const labelMap = {
  // Color
  chlorophyll: 'Overall Green Color (Chlorophyll)',
  chlorosis: 'Yellow Discoloration (Chlorosis)',
  necrosis: 'Brown Dead Tissue (Necrosis)',
  interveinal: 'Yellowing Between Veins',
  marginal: 'Brown Leaf Edges',
  
  // Structure
  leafSize: 'Leaf Size',
  shapeDistortion: 'Leaf Shape',
  edgeIrreg: 'Edge Pattern',
  
  // Diseases
  leafSpot: 'Leaf Spot Disease (Fungal)',
  powderyMildew: 'Powdery Mildew',
  anthracnose: 'Anthracnose',
  angularSpot: 'Angular Leaf Spot (Bacterial)',
  
  // Nutrients
  nitrogen: 'Nitrogen (N)',
  potassium: 'Potassium (K)',
  iron: 'Iron (Fe)',
  magnesium: 'Magnesium (Mg)',
  
  // Physical
  chewing: 'Chewing Damage (Caterpillars/Beetles)',
  stippling: 'Yellow Dots (Spider Mites)',
  
  // Environmental
  sunburn: 'Sunburn/Heat Stress',
  waterStress: 'Water Stress (Wilting)'
};

// Explanation Modal Component
const ExplanationModal = ({ itemKey, result, onClose }) => {
  if (!result) return null;
  
  const isHealthy = result.isHealthy === true || result.isHealthy === 1;
  const StatusIcon = isHealthy ? CheckCircle : AlertTriangle;
  const statusColor = isHealthy ? '#4caf50' : '#ff9800';
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      background: 'rgba(0, 0, 0, 0.85)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '16px',
      backdropFilter: 'blur(8px)'
    }} onClick={onClose}>
      <div 
        style={{
          background: 'linear-gradient(145deg, #1a3a1a 0%, #0d2d0d 100%)',
          borderRadius: 20,
          maxWidth: 600,
          width: '100%',
          maxHeight: '90vh',
          overflow: 'auto',
          border: '2px solid rgba(144, 238, 144, 0.3)',
          boxShadow: '0 20px 60px rgba(0, 0, 0, 0.5)'
        }}
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '1px solid rgba(144, 238, 144, 0.2)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          background: 'linear-gradient(90deg, rgba(45, 90, 39, 0.3) 0%, transparent 100%)'
        }}>
          <div style={{ flex: 1, paddingRight: '12px' }}>
            <h3 style={{ margin: 0, color: '#90EE90', fontSize: '1.1rem', fontWeight: 700, lineHeight: 1.3 }}>
              {labelMap[itemKey] || itemKey}
            </h3>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 8 }}>
              <StatusIcon size={18} color={statusColor} />
              <span style={{ color: statusColor, fontWeight: 600, fontSize: '0.95rem' }}>
                {result.value}
              </span>
            </div>
          </div>
          <button onClick={onClose} style={{
            background: 'rgba(255,255,255,0.1)', border: 'none', borderRadius: 8,
            width: 36, height: 36, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0
          }}>
            <X size={20} color="#90EE90" />
          </button>
        </div>
        
        {/* Confidence Score */}
        <div style={{ padding: '14px 20px', borderBottom: '1px solid rgba(144, 238, 144, 0.1)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ color: '#a5d6a7', fontSize: '0.85rem' }}>AI Confidence Level</span>
            <span style={{ color: '#90EE90', fontWeight: 700 }}>{Math.round(result.confidence || 0)}%</span>
          </div>
          <div style={{ height: 8, background: 'rgba(144, 238, 144, 0.15)', borderRadius: 4, overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${result.confidence || 0}%`,
              background: (result.confidence || 0) > 85 
                ? 'linear-gradient(90deg, #4caf50, #8bc34a)' 
                : (result.confidence || 0) > 70 
                  ? 'linear-gradient(90deg, #ff9800, #ffc107)'
                  : 'linear-gradient(90deg, #f44336, #ff5722)',
              borderRadius: 4,
              transition: 'width 0.5s ease'
            }} />
          </div>
        </div>
        
        {/* Explanation */}
        <div style={{ padding: '16px 20px' }}>
          <div style={{ marginBottom: 20 }}>
            <h4 style={{
              color: '#90EE90', fontSize: '0.85rem', fontWeight: 700, marginBottom: 10,
              textTransform: 'uppercase', letterSpacing: '0.5px', display: 'flex', alignItems: 'center', gap: 8
            }}>
              <Info size={16} />
              What This Means
            </h4>
            <p style={{ color: '#c8e6c9', fontSize: '0.9rem', lineHeight: 1.6, margin: 0 }}>
              {result.explanation || 'Analysis completed based on computer vision algorithms.'}
            </p>
          </div>
          
          {/* Evidence */}
          {result.evidence && result.evidence.length > 0 && (
            <div style={{ marginBottom: 20 }}>
              <h4 style={{
                color: '#90EE90', fontSize: '0.85rem', fontWeight: 700, marginBottom: 12,
                textTransform: 'uppercase', letterSpacing: '0.5px'
              }}>
                📊 Detection Details
              </h4>
              <ul style={{ margin: 0, padding: 0, listStyle: 'none' }}>
                {result.evidence.map((item, idx) => (
                  <li key={idx} style={{
                    display: 'flex', alignItems: 'flex-start', gap: 10, padding: '8px 12px',
                    background: 'rgba(144, 238, 144, 0.08)', borderRadius: 8, marginBottom: 8,
                    borderLeft: '3px solid rgba(144, 238, 144, 0.4)'
                  }}>
                    <span style={{ color: '#69f0ae', fontSize: '0.8rem' }}>•</span>
                    <span style={{ color: '#a5d6a7', fontSize: '0.85rem', lineHeight: 1.5 }}>{item}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          
          {/* Recommendation */}
          <div style={{
            padding: 14,
            background: isHealthy 
              ? 'linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.05) 100%)'
              : 'linear-gradient(135deg, rgba(255, 152, 0, 0.15) 0%, rgba(255, 152, 0, 0.05) 100%)',
            borderRadius: 12,
            border: `1px solid ${isHealthy ? 'rgba(76, 175, 80, 0.3)' : 'rgba(255, 152, 0, 0.3)'}`
          }}>
            <h4 style={{
              color: isHealthy ? '#81c784' : '#ffb74d', fontSize: '0.85rem', fontWeight: 700,
              marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.5px'
            }}>
              💡 Action Plan
            </h4>
            <p style={{ color: isHealthy ? '#a5d6a7' : '#ffe0b2', fontSize: '0.85rem', lineHeight: 1.6, margin: 0 }}>
              {result.recommendation || 'Continue monitoring the plant health.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

const CategoryCard = ({ category, results, expanded, onToggle, onItemClick }) => {
  return (
    <div style={{
      background: 'linear-gradient(145deg, rgba(45, 90, 39, 0.15) 0%, rgba(30, 60, 26, 0.25) 100%)',
      borderRadius: 12,
      border: '1px solid rgba(45, 90, 39, 0.3)',
      overflow: 'hidden',
      backdropFilter: 'blur(10px)'
    }}>
      <button onClick={onToggle} style={{
        width: '100%', padding: '12px 16px',
        background: 'linear-gradient(90deg, rgba(45, 90, 39, 0.4) 0%, rgba(45, 90, 39, 0.2) 100%)',
        border: 'none', borderBottom: '1px solid rgba(45, 90, 39, 0.3)',
        cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center'
      }}>
        <span style={{
          color: '#90EE90', fontSize: '0.8rem', fontWeight: 700, letterSpacing: '0.5px',
          textDecoration: 'underline', textUnderlineOffset: '3px'
        }}>
          {category.title}
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#90EE90', fontSize: '0.7rem' }}>
          <span className="hide-on-mobile">CLICK FOR DETAILS</span>
          <ChevronDown size={16} style={{ transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s ease' }} />
        </div>
      </button>

      {expanded && (
        <div style={{ padding: '12px 16px' }}>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: 4 }}>
            {category.keys.map((key) => {
              const result = results ? results[key] : null;
              const isHealthy = result ? (result.isHealthy === true || result.isHealthy === 1) : true;
              
              return (
                <li 
                  key={key}
                  onClick={() => result && onItemClick(key)}
                  style={{
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    padding: '8px 10px', borderRadius: 6, cursor: result ? 'pointer' : 'default',
                    transition: 'all 0.2s ease', background: 'transparent', border: '1px solid transparent'
                  }}
                  onMouseEnter={(e) => { if (result) { e.currentTarget.style.background = 'rgba(144, 238, 144, 0.1)'; e.currentTarget.style.borderColor = 'rgba(144, 238, 144, 0.2)'; }}}
                  onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.borderColor = 'transparent'; }}
                >
                  <span style={{ color: '#c8e6c9', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ color: '#90EE90' }}>•</span>
                    {labelMap[key] || key}:
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    {result ? (
                      <>
                        <span style={{
                          color: isHealthy ? '#90EE90' : '#ffab91',
                          fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px',
                          background: isHealthy ? 'rgba(144, 238, 144, 0.15)' : 'rgba(255, 171, 145, 0.15)',
                          borderRadius: 4, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap'
                        }}>
                          {result.value || '—'}
                        </span>
                        <Info size={14} color="rgba(144, 238, 144, 0.6)" style={{ flexShrink: 0 }} />
                      </>
                    ) : (
                      <span style={{ color: 'rgba(144, 238, 144, 0.4)', fontSize: '0.75rem' }}>—</span>
                    )}
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

function App() {
  const [image, setImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisData, setAnalysisData] = useState(null);
  const [error, setError] = useState(null);
  const [expandedCards, setExpandedCards] = useState({});
  const [selectedItem, setSelectedItem] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      // Check file size (limit to 5MB)
      if (file.size > 5 * 1024 * 1024) {
        setError('Image too large! Please use an image smaller than 5MB.');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = (e) => {
        const base64 = e.target.result;
        
        // Check base64 size (limit to 7MB encoded)
        if (base64.length > 7 * 1024 * 1024) {
          setError('Image too large after encoding! Please compress the image or reduce its quality.');
          return;
        }
        
        setImage(base64);
        setAnalysisData(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!image) return;
    
    setIsAnalyzing(true);
    setError(null);
    setExpandedCards({});
    
    try {
      // Add timeout to prevent hanging requests
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 second timeout (increased for Render free tier cold start)
      
      const response = await fetch("https://strawberry-analyzer-rt5w.onrender.com/api/analyze", {
        method: "POST",
        headers: { 
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          image: image
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Server response:', errorText);
        throw new Error(`Server error ${response.status}: ${response.statusText}`);
      }

      const analysisResult = await response.json();
      
      if (!analysisResult.success) {
        throw new Error(analysisResult.error || 'Analysis failed');
      }
      
      setAnalysisData(analysisResult);
      setExpandedCards({ colorBased: true });
    } catch (err) {
      console.error('Analysis error:', err);
      if (err.name === 'AbortError') {
        setError('Request timed out. The server might be starting up (can take 30-50 seconds on first use). Please try again.');
      } else if (err.message.includes('Failed to fetch')) {
        setError('Cannot connect to server. The server might be waking up (takes 30-50s on first use). Please wait and try again.');
      } else {
        setError(`Failed to analyze: ${err.message}`);
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const toggleCard = (key) => {
    setExpandedCards(prev => ({ ...prev, [key]: !prev[key] }));
  };

  const handleItemClick = (key) => {
    setSelectedItem(key);
  };

  const results = analysisData?.results || {};
  const healthScore = analysisData?.overall_score || 0;

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a1f0a 0%, #1a3a1a 30%, #0d2d0d 70%, #051405 100%)',
      fontFamily: "'Segoe UI', system-ui, sans-serif",
      padding: '16px',
      position: 'relative'
    }}>
      {/* Background texture */}
      <div style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        backgroundImage: 'radial-gradient(circle at 2px 2px, rgba(144, 238, 144, 0.03) 1px, transparent 0)',
        backgroundSize: '32px 32px', pointerEvents: 'none'
      }} />

      <div style={{ maxWidth: 1400, margin: '0 auto', position: 'relative' }}>
        {/* Header */}
        <header style={{
          textAlign: 'center', marginBottom: 24, padding: '16px', position: 'relative',
          background: 'linear-gradient(90deg, transparent, rgba(45, 90, 39, 0.3), transparent)', borderRadius: 16
        }}>
          <div className="lsu-logo" style={{
            position: 'absolute',
            top: '16px',
            right: '16px',
            width: '100px',
            height: 'auto',
            zIndex: 10
          }}>
            <img 
              src={LSU_LOGO} 
              alt="LSU AgCenter" 
              style={{
                width: '100%',
                height: 'auto',
                display: 'block',
                filter: 'drop-shadow(0 2px 8px rgba(0, 0, 0, 0.5))'
              }}
            />
          </div>
          
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: 12, marginBottom: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
            <span style={{ fontSize: '2rem' }}>🍓</span>
            <h1 className="main-title" style={{
              fontSize: '1.5rem', fontWeight: 800, color: '#90EE90', margin: 0,
              textTransform: 'uppercase', letterSpacing: '1.5px', textShadow: '0 0 20px rgba(144, 238, 144, 0.5)'
            }}>
              Strawberry Leaf Health Analyzer
            </h1>
          </div>
          <p className="subtitle" style={{ color: '#69f0ae', fontSize: '0.8rem', margin: '8px 0 0 0', opacity: 0.8, paddingRight: '120px' }}>
            AI-Powered Detection of Key Issues • Phone Camera Optimized
          </p>
        </header>

        {/* Main Grid */}
        <div className="main-grid" style={{ display: 'grid', gridTemplateColumns: '280px 1fr 1fr', gap: 16, alignItems: 'start' }}>
          {/* Left Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {/* Health Score */}
            <div style={{
              background: 'linear-gradient(145deg, rgba(45, 90, 39, 0.2) 0%, rgba(30, 60, 26, 0.3) 100%)',
              borderRadius: 16, padding: 16, border: '1px solid rgba(45, 90, 39, 0.4)', textAlign: 'center'
            }}>
              <div style={{ color: '#90EE90', fontSize: '0.9rem', fontWeight: 700, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '1px' }}>
                Health Score
              </div>
              
              {/* Gauge */}
              <div style={{ position: 'relative', height: 90, marginBottom: 8 }}>
                <svg viewBox="0 0 200 100" style={{ width: '100%', height: '100%' }}>
                  <path d="M 20 90 A 70 70 0 0 1 180 90" fill="none" stroke="rgba(144, 238, 144, 0.2)" strokeWidth="12" strokeLinecap="round" />
                  <defs>
                    <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#ff5252" />
                      <stop offset="25%" stopColor="#ff9800" />
                      <stop offset="50%" stopColor="#ffeb3b" />
                      <stop offset="75%" stopColor="#8bc34a" />
                      <stop offset="100%" stopColor="#4caf50" />
                    </linearGradient>
                  </defs>
                  <path d="M 20 90 A 70 70 0 0 1 180 90" fill="none" stroke="url(#gaugeGradient)" strokeWidth="12" strokeLinecap="round"
                    strokeDasharray={`${(analysisData ? healthScore : 0) * 2.2} 220`} style={{ transition: 'stroke-dasharray 1s ease-out' }} />
                  <g transform={`rotate(${-90 + (analysisData ? healthScore : 0) * 1.8}, 100, 90)`}>
                    <polygon points="100,30 95,85 105,85" fill="#fff" style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.3))' }} />
                    <circle cx="100" cy="90" r="8" fill="#2d5a27" stroke="#90EE90" strokeWidth="2" />
                  </g>
                </svg>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0 10px', marginBottom: 12 }}>
                <span style={{ color: '#ff8a80', fontSize: '1.1rem', fontWeight: 700 }}>0</span>
                <span style={{ color: '#90EE90', fontSize: '1.8rem', fontWeight: 800, textShadow: '0 0 10px rgba(144, 238, 144, 0.5)' }}>
                  {analysisData ? Math.round(healthScore) : '—'}
                </span>
                <span style={{ color: '#69f0ae', fontSize: '1.1rem', fontWeight: 700 }}>100</span>
              </div>
              
              {/* Status Badge */}
              {analysisData && (
                <div style={{
                  padding: '8px 16px',
                  borderRadius: 20,
                  background: healthScore >= 70 ? 'rgba(76, 175, 80, 0.2)' : healthScore >= 50 ? 'rgba(255, 193, 7, 0.2)' : 'rgba(244, 67, 54, 0.2)',
                  color: healthScore >= 70 ? '#81c784' : healthScore >= 50 ? '#ffd54f' : '#e57373',
                  fontSize: '0.8rem',
                  fontWeight: 600
                }}>
                  {healthScore >= 70 ? '✓ Healthy' : healthScore >= 50 ? '⚠ Moderate' : '✗ Needs Attention'}
                </div>
              )}
            </div>

            {/* Upload Section */}
            <div style={{
              background: 'linear-gradient(145deg, rgba(45, 90, 39, 0.2) 0%, rgba(30, 60, 26, 0.3) 100%)',
              borderRadius: 16, padding: 16, border: '1px solid rgba(45, 90, 39, 0.4)'
            }}>
              <div style={{ color: '#90EE90', fontSize: '0.9rem', fontWeight: 700, marginBottom: 12, textTransform: 'uppercase', letterSpacing: '1px' }}>
                Upload Leaf Image
              </div>

              <div style={{
                width: '100%', aspectRatio: '1', borderRadius: 12, overflow: 'hidden', marginBottom: 12,
                border: '2px solid rgba(45, 90, 39, 0.5)', background: 'rgba(0, 20, 0, 0.5)', position: 'relative'
              }}>
                {image ? (
                  <>
                    <img src={image} alt="Uploaded leaf" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                    <button onClick={() => { setImage(null); setAnalysisData(null); setError(null); }}
                      style={{
                        position: 'absolute', top: 8, right: 8, width: 28, height: 28, borderRadius: '50%',
                        background: 'rgba(0,0,0,0.6)', border: 'none', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                      }}>
                      <X size={16} color="#fff" />
                    </button>
                  </>
                ) : (
                  <div style={{
                    width: '100%', height: '100%', display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', color: '#5a8a5a'
                  }}>
                    <Leaf size={48} style={{ opacity: 0.5, marginBottom: 8 }} />
                    <span style={{ fontSize: '0.75rem' }}>No image uploaded</span>
                  </div>
                )}
              </div>

              <button onClick={() => fileInputRef.current?.click()} style={{
                width: '100%', padding: '12px 16px',
                background: 'linear-gradient(135deg, #2d5a27 0%, #1a3a15 100%)',
                border: '2px solid #90EE90', borderRadius: 8, color: '#90EE90',
                fontSize: '0.8rem', fontWeight: 700, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 12
              }}>
                <Upload size={18} />
                {image ? 'Change Image' : 'Upload Image'}
              </button>

              {image && (
                <button onClick={analyzeImage} disabled={isAnalyzing} style={{
                  width: '100%', padding: '14px 16px',
                  background: isAnalyzing ? 'rgba(144, 238, 144, 0.2)' : 'linear-gradient(135deg, #4caf50 0%, #2e7d32 100%)',
                  border: 'none', borderRadius: 8, color: '#fff',
                  fontSize: '0.85rem', fontWeight: 700, cursor: isAnalyzing ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  textTransform: 'uppercase', letterSpacing: '0.5px',
                  boxShadow: isAnalyzing ? 'none' : '0 4px 15px rgba(76, 175, 80, 0.4)'
                }}>
                  {isAnalyzing ? (<><Loader2 size={18} className="spin" style={{ animation: 'spin 1s linear infinite' }} /> Analyzing... (First use: 30-50s)</>) 
                    : (<><Camera size={18} /> Analyze Leaf</>)}
                </button>
              )}

              <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} style={{ display: 'none' }} />
            </div>

            {/* Error Display */}
            {error && (
              <div style={{
                padding: 14, background: 'rgba(244, 67, 54, 0.15)', borderRadius: 12,
                border: '1px solid rgba(244, 67, 54, 0.3)', color: '#ff8a80', fontSize: '0.8rem', lineHeight: 1.4
              }}>
                ⚠️ {error}
              </div>
            )}

            {/* Info Box */}
            <div className="info-section" style={{
              padding: '14px', background: 'linear-gradient(145deg, rgba(45, 90, 39, 0.15) 0%, rgba(30, 60, 26, 0.2) 100%)',
              borderRadius: 12, border: '1px solid rgba(45, 90, 39, 0.3)'
            }}>
              <div style={{ color: '#90EE90', fontSize: '0.75rem', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase' }}>
                📱 Phone Camera Tips
              </div>
              <ul style={{ color: '#a5d6a7', fontSize: '0.7rem', lineHeight: 1.5, margin: 0, paddingLeft: 16 }}>
                <li>Use natural daylight</li>
                <li>Hold camera 6-8 inches away</li>
                <li>Focus on single leaf</li>
                <li>Avoid shadows</li>
                <li>Keep image under 5MB</li>
                <li>First use may take 30-50s</li>
              </ul>
            </div>

            {/* Summary Stats */}
            {analysisData && (
              <div className="summary-section" style={{
                padding: '14px', background: 'linear-gradient(145deg, rgba(45, 90, 39, 0.15) 0%, rgba(30, 60, 26, 0.2) 100%)',
                borderRadius: 12, border: '1px solid rgba(45, 90, 39, 0.3)'
              }}>
                <div style={{ color: '#90EE90', fontSize: '0.75rem', fontWeight: 600, marginBottom: 10, textTransform: 'uppercase' }}>
                  Analysis Summary
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#a5d6a7', fontSize: '0.8rem' }}>
                  <span>Healthy Indicators:</span>
                  <span style={{ color: '#90EE90', fontWeight: 600 }}>{analysisData.healthy_count || 0}/{analysisData.total_count || 0}</span>
                </div>
              </div>
            )}
          </div>

          {/* Middle Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <CategoryCard category={healthCategories.colorBased} results={results} expanded={expandedCards.colorBased} onToggle={() => toggleCard('colorBased')} onItemClick={handleItemClick} />
            <CategoryCard category={healthCategories.structural} results={results} expanded={expandedCards.structural} onToggle={() => toggleCard('structural')} onItemClick={handleItemClick} />
            <CategoryCard category={healthCategories.nutrient} results={results} expanded={expandedCards.nutrient} onToggle={() => toggleCard('nutrient')} onItemClick={handleItemClick} />
          </div>

          {/* Right Column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <CategoryCard category={healthCategories.disease} results={results} expanded={expandedCards.disease} onToggle={() => toggleCard('disease')} onItemClick={handleItemClick} />
            <CategoryCard category={healthCategories.physical} results={results} expanded={expandedCards.physical} onToggle={() => toggleCard('physical')} onItemClick={handleItemClick} />
            <CategoryCard category={healthCategories.environmental} results={results} expanded={expandedCards.environmental} onToggle={() => toggleCard('environmental')} onItemClick={handleItemClick} />
          </div>
        </div>

        {/* Footer */}
        <footer style={{
          textAlign: 'center', marginTop: 32, padding: 16, color: '#5a8a5a', fontSize: '0.75rem'
        }}>
          All rights reserved @LSUAgCenter • Made For 🍓 Farmers and Researchers.
        </footer>
      </div>

      {/* Explanation Modal */}
      {selectedItem && results[selectedItem] && (
        <ExplanationModal itemKey={selectedItem} result={results[selectedItem]} onClose={() => setSelectedItem(null)} />
      )}

      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: rgba(45, 90, 39, 0.2); }
        ::-webkit-scrollbar-thumb { background: rgba(144, 238, 144, 0.4); border-radius: 3px; }
        
        @media (max-width: 1024px) {
          .main-grid {
            grid-template-columns: 1fr !important;
          }
          
          .lsu-logo {
            width: 80px !important;
            top: 12px !important;
            right: 12px !important;
          }
          
          .main-title {
            font-size: 1.3rem !important;
            letter-spacing: 1px !important;
          }
          
          .subtitle {
            font-size: 0.75rem !important;
            padding-right: 90px !important;
          }
        }
        
        @media (max-width: 768px) {
          .main-grid {
            gap: 12px !important;
          }
          
          .lsu-logo {
            width: 70px !important;
            top: 8px !important;
            right: 8px !important;
          }
          
          .main-title {
            font-size: 1.1rem !important;
            text-align: left !important;
          }
          
          .subtitle {
            font-size: 0.7rem !important;
            padding-right: 80px !important;
            text-align: left !important;
          }
          
          .hide-on-mobile {
            display: none !important;
          }
          
          .info-section,
          .summary-section {
            font-size: 0.7rem !important;
          }
        }
        
        @media (max-width: 480px) {
          .lsu-logo {
            width: 60px !important;
          }
          
          .main-title {
            font-size: 1rem !important;
          }
          
          .subtitle {
            font-size: 0.65rem !important;
          }
        }
      `}</style>
    </div>
  );
}

export default App;
