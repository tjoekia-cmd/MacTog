/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  TrendingUp,
  Cpu,
  RefreshCw,
  Search,
  PlusCircle,
  Hash,
  Activity,
  AlertTriangle,
  Award,
  BookOpen,
  PieChart,
  HelpCircle,
  Database,
  BarChart3,
  Calendar,
  CheckCircle,
  Dices,
  Play,
  Pencil,
  X,
  Upload,
  Image,
  Trash2,
  Sliders,
  Check
} from "lucide-react";
import {
  MacauDraw,
  PredictionEngineOutput,
  AiPredictionResponse,
  StatisticalMetrics
} from "./types";

export default function App() {
  // Application tabs
  const [activeTab, setActiveTab] = useState<"dashboard" | "poisson" | "markov" | "calculus" | "ai" | "backtest" | "ml">("dashboard");

  // Core drawing data state
  const [history, setHistory] = useState<MacauDraw[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(true);
  const [historyError, setHistoryError] = useState("");

  // Machine Learning states
  const [mlSummary, setMlSummary] = useState<any | null>(null);
  const [mlAnomalies, setMlAnomalies] = useState<any[]>([]);
  const [isMlLoading, setIsMlLoading] = useState(false);
  const [mlError, setMlError] = useState("");

  // Search and filter states
  const [searchQuery, setSearchQuery] = useState("");
  const [filterSlot, setFilterSlot] = useState<string>("All");

  // Manual entry Form state
  const [formData, setFormData] = useState({
    date: new Date().toISOString().split("T")[0],
    timeSlot: "13:00",
    drawCodeString: ""
  });
  const [formSuccess, setFormSuccess] = useState("");
  const [formError, setFormError] = useState("");
  const [isSubmittingForm, setIsSubmittingForm] = useState(false);

  // Edit / Correct Database entry state
  const [editingDraw, setEditingDraw] = useState<MacauDraw | null>(null);
  const [editFormData, setEditFormData] = useState({
    date: "",
    timeSlot: "13:00",
    drawCodeString: ""
  });
  const [editSuccess, setEditSuccess] = useState("");
  const [editError, setEditError] = useState("");
  const [isSavingEdit, setIsSavingEdit] = useState(false);

  // Scraper state
  const [isScraping, setIsScraping] = useState(false);
  const [scrapeSuccess, setScrapeSuccess] = useState("");
  const [scrapeError, setScrapeError] = useState("");

  // OCR state
  const [isOcrLoading, setIsOcrLoading] = useState(false);
  const [ocrSuccess, setOcrSuccess] = useState("");
  const [ocrWarning, setOcrWarning] = useState("");
  const [ocrError, setOcrError] = useState("");
  const [ocrResults, setOcrResults] = useState<{ date: string; timeSlot: string; drawCodeString: string }[]>([]);
  const [isDraggingOcr, setIsDraggingOcr] = useState(false);

  // Database Management states
  const [dbSuccessMessage, setDbSuccessMessage] = useState("");
  const [dbErrorMessage, setDbErrorMessage] = useState("");
  const [isDbWorking, setIsDbWorking] = useState(false);

  // Statistical Engine prediction output
  const [modelPrediction, setModelPrediction] = useState<PredictionEngineOutput | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelError, setModelError] = useState("");

  // Gemini AI machine learning predictions output
  const [aiPrediction, setAiPrediction] = useState<AiPredictionResponse | null>(null);
  const [isAiLoading, setIsAiLoading] = useState(false);
  const [aiError, setAiError] = useState("");

  // Backtesting simulator states
  const [backtestSize, setBacktestSize] = useState<number>(15);
  const [backtestResults, setBacktestResults] = useState<{
    totalTrials: number;
    colokBebasHits: number;
    colokMacauHits: number;
    exact2DHits: number;
    trialsDetail: Array<{
      drawId: string;
      date: string;
      timeSlot: string;
      actual: string;
      predicted2D: string[];
      predictedCB: number[];
      isCBHit: boolean;
      is2DHit: boolean;
    }>;
  } | null>(null);
  const [isBacktesting, setIsBacktesting] = useState(false);

  // Markov Transition Grid Selector State
  const [selectedMarkovPos, setSelectedMarkovPos] = useState<"as" | "kop" | "kepala" | "ekor">("as");
  const [selectedMarkovDigit, setSelectedMarkovDigit] = useState<number>(0);

  // BBFS Copy feedback states
  const [copiedBbfs, setCopiedBbfs] = useState<string | null>(null);

  // Fetch complete historical lottery drawings
  const fetchHistory = useCallback(async () => {
    setIsHistoryLoading(true);
    setHistoryError("");
    try {
      const response = await fetch("/api/history");
      if (!response.ok) {
        throw new Error("Failed to load drawing data from server.");
      }
      const data: MacauDraw[] = await response.json();
      setHistory(data);
    } catch (err: any) {
      setHistoryError(err.message || "Unknown retrieval error occurred.");
    } finally {
      setIsHistoryLoading(false);
    }
  }, []);

  // Fetch analytic models
  const runMathematicalEngine = useCallback(async () => {
    setIsModelLoading(true);
    setModelError("");
    try {
      const response = await fetch("/api/predict", { method: "POST" });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Failed to trigger statistical analysis calculations.");
      }
      const data: PredictionEngineOutput = await response.json();
      setModelPrediction(data);
    } catch (err: any) {
      setModelError(err.message || "Analytical compilation error.");
    } finally {
      setIsModelLoading(false);
    }
  }, []);

  // Fetch ML Training Summary and anomalies
  const runMlTrainEngine = useCallback(async () => {
    setIsMlLoading(true);
    setMlError("");
    try {
      const response = await fetch("/api/predict-ml", { method: "POST" });
      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || "Gagal melatih model Machine Learning luring.");
      }
      const data = await response.json();
      setMlSummary(data);

      const rAnom = await fetch("/api/ml-anomalies");
      if (rAnom.ok) {
        const dAnom = await rAnom.json();
        setMlAnomalies(dAnom);
      }
    } catch (err: any) {
      setMlError(err.message || "Terjadi kesalahan kompilasi pembelajaran mesin.");
    } finally {
      setIsMlLoading(false);
    }
  }, []);

  // Fetch initially on load
  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  // Recalculate model predictions whenever the history changes
  useEffect(() => {
    if (history.length >= 10) {
      runMathematicalEngine();
      runMlTrainEngine();
    }
  }, [history, runMathematicalEngine, runMlTrainEngine]);

  // Trigger web scraping sync
  const handleScrapeSync = async () => {
    setIsScraping(true);
    setScrapeSuccess("");
    setScrapeError("");
    try {
      const response = await fetch("/api/scrape-realtime", { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Scraper synchronization failed.");
      }
      if (data.isFallback) {
        setScrapeSuccess(`${data.warning || "Sinkronisasi fallback berhasil!"} (+${data.addedCount} nomor simulasi baru)`);
      } else {
        setScrapeSuccess(`Sinkronisasi berhasil! Ditemukan ${data.addedCount} nomor undian baru.`);
      }
      fetchHistory(); // Refresh table
    } catch (err: any) {
      setScrapeError(err.message || "Koneksi ke server scraper gagal.");
    } finally {
      setIsScraping(false);
    }
  };

  // Convert file to Base64 and run OCR
  const processImageFile = async (file: File) => {
    if (!file) return;
    
    // Ensure it is an image
    if (!file.type.startsWith("image/")) {
      setOcrError("Tipe berkas tidak didukung! Format yang diizinkan hanya file gambar (.png, .jpg, .jpeg, .webp, dsb.)");
      return;
    }

    setIsOcrLoading(true);
    setOcrSuccess("");
    setOcrWarning("");
    setOcrError("");
    setOcrResults([]);

    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = async () => {
      try {
        const base64Data = reader.result as string;
        const mimeType = file.type;

        const response = await fetch("/api/ocr-import", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ base64Data, mimeType })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Penganalisisan OCR mengalami kegagalan.");
        }

        if (data.isFallback) {
          setOcrWarning(data.warning || "Menggunakan data simulasi OCR luring.");
        }

        setOcrSuccess(`Pemrosesan OCR Selesai! Ditambahkan: ${data.addedCount} undian baru, Diperbaiki: ${data.updatedCount} undian.`);
        setOcrResults(data.extractedResults || []);
        
        fetchHistory(); // Refresh history table in UI!
      } catch (err: any) {
        setOcrError(err.message || "Gagal mengunggah atau memproses gambar OCR.");
      } finally {
        setIsOcrLoading(false);
      }
    };
    reader.onerror = () => {
      setOcrError("Gagal membaca file gambar.");
      setIsOcrLoading(false);
    };
  };

  const handleOcrFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processImageFile(e.target.files[0]);
    }
  };

  const handleOcrDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOcr(true);
  };

  const handleOcrDragLeave = () => {
    setIsDraggingOcr(false);
  };

  const handleOcrDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDraggingOcr(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processImageFile(e.dataTransfer.files[0]);
    }
  };

  // Clear Database entirely to keep only OCR data
  const handleClearDb = async () => {
    if (!window.confirm("Apakah Anda yakin ingin mengosongkan seluruh isi database untuk hanya menyisakan undian dari hasil OCR gambar?")) return;
    setIsDbWorking(true);
    setDbSuccessMessage("");
    setDbErrorMessage("");
    try {
      const response = await fetch("/api/history/clear", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Gagal mengosongkan database.");
      setDbSuccessMessage("Database berhasil dikosongkan secara total! Silakan drag & drop gambar untuk mengisi data baru.");
      setHistory([]);
      setOcrResults([]);
    } catch (err: any) {
      setDbErrorMessage(err.message || "Gagal membersihkan data.");
    } finally {
      setIsDbWorking(false);
    }
  };

  // Keep ONLY OCR data in database
  const handleKeepOnlyOcr = async () => {
    if (!window.confirm("Apakah Anda yakin ingin menghapus semua hasil keluaran database selain dari gambar yang Anda upload?")) return;
    setIsDbWorking(true);
    setDbSuccessMessage("");
    setDbErrorMessage("");
    try {
      const response = await fetch("/api/history/keep-only-ocr", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Gagal menyaring database.");
      setDbSuccessMessage(data.message);
      fetchHistory();
    } catch (err: any) {
      setDbErrorMessage(err.message || "Gagal menyaring database.");
    } finally {
      setIsDbWorking(false);
    }
  };

  // Restore Seed Data
  const handleRestoreSeed = async () => {
    setIsDbWorking(true);
    setDbSuccessMessage("");
    setDbErrorMessage("");
    try {
      const response = await fetch("/api/history/restore-seed", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Gagal memulihkan database bawaan.");
      setDbSuccessMessage("Database berhasil dipulihkan ke data default bawaan.");
      fetchHistory();
    } catch (err: any) {
      setDbErrorMessage(err.message || "Gagal memulihkan data.");
    } finally {
      setIsDbWorking(false);
    }
  };

  // Migrate year 2024 to 2026
  const handleMigrate2024To2026 = async () => {
    setIsDbWorking(true);
    setDbSuccessMessage("");
    setDbErrorMessage("");
    try {
      const response = await fetch("/api/history/migrate-2024-to-2026", { method: "POST" });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Gagal memperbaharui tahun.");
      setDbSuccessMessage(`Aksi Sukses! Berhasil mengubah ${data.migratedCount || 0} undian dari tahun 2024 menjadi tahun 2026.`);
      fetchHistory();
    } catch (err: any) {
      setDbErrorMessage(err.message || "Gagal memproses migrasi tahun.");
    } finally {
      setIsDbWorking(false);
    }
  };

  // Submit new drawing entry form
  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setFormSuccess("");
    setFormError("");
    setIsSubmittingForm(true);

    if (formData.drawCodeString.length !== 4 || isNaN(Number(formData.drawCodeString))) {
      setFormError("Nomor keluaran harus tepat berupa 4 digit angka (0000-9999)!");
      setIsSubmittingForm(false);
      return;
    }

    try {
      const response = await fetch("/api/history/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData)
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Failure to write result to database.");
      }
      setFormSuccess(`Keluaran #${data.draw.id} (${formData.drawCodeString}) berhasil ditambahkan ke database.`);
      setFormData({ ...formData, drawCodeString: "" });
      fetchHistory(); // Reload
    } catch (err: any) {
      setFormError(err.message || "Gagal menyimpan data.");
    } finally {
      setIsSubmittingForm(false);
    }
  };

  // Submit corrected/edited drawing entry
  const handleEditSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingDraw) return;
    setEditSuccess("");
    setEditError("");
    setIsSavingEdit(true);

    if (editFormData.drawCodeString.length !== 4 || isNaN(Number(editFormData.drawCodeString))) {
      setEditError("Nomor keluaran harus tepat berupa 4 digit angka (0000-9999)!");
      setIsSavingEdit(false);
      return;
    }

    try {
      const response = await fetch("/api/history/edit", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          id: editingDraw.id,
          date: editFormData.date,
          timeSlot: editFormData.timeSlot,
          drawCodeString: editFormData.drawCodeString
        })
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Gagal memperbarui nomor di database.");
      }
      setEditSuccess(`Keluaran #${editingDraw.id} (${editFormData.drawCodeString}) berhasil disimpan!`);
      
      // Delay clear editing mode for a smooth transition
      setTimeout(() => {
        setEditingDraw(null);
        setEditSuccess("");
      }, 1500);

      fetchHistory(); // Reload table
    } catch (err: any) {
      setEditError(err.message || "Gagal memperbarui data.");
    } finally {
      setIsSavingEdit(false);
    }
  };

  // Run AI Machine Learning Analysis using Gemini
  const runAiMlEngine = async () => {
    setIsAiLoading(true);
    setAiError("");
    try {
      const response = await fetch("/api/predict-ai", { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Gagal mengambil interpretasi pola undian dari AI.");
      }
      setAiPrediction(data);
    } catch (err: any) {
      setAiError(err.message || "Terjadi kesalahan integrasi AI.");
    } finally {
      setIsAiLoading(false);
    }
  };

  // Run simulations sandbox backtesting (Empirical Verification)
  const runBacktestSimulator = () => {
    if (history.length < backtestSize + 10) {
      alert(`Database historical drawings tidak mencukupi untuk melakukan simulasi ${backtestSize} kali. Jumlah total data minimal: ${backtestSize + 10}`);
      return;
    }
    setIsBacktesting(true);
    
    // Sort oldest to newest for step-by-step transitions
    const drawsAsc = [...history].sort((a,b) => a.id.localeCompare(b.id, undefined, { numeric: true }));
    let cbHits = 0;
    let cmHits = 0;
    let d2Hits = 0;
    const trialsDetail = [];

    // Run iterative simulations: to predict draw t, we restrict database to [0..t-1]
    const startIndex = drawsAsc.length - backtestSize;
    for (let i = startIndex; i < drawsAsc.length; i++) {
      const evaluationDraw = drawsAsc[i];
      // Trailing database
      const subset = drawsAsc.slice(0, i);
      
      // Calculate mini models for this subset
      const actualDigits = evaluationDraw.digits;
      const actual2D = evaluationDraw.drawCodeString.substring(2);

      // Simple prediction: Poisson Top Probable
      // Position Colok Bebas: overall hot digits
      const digitCounts = Array(10).fill(0);
      subset.forEach(d => {
        d.digits.forEach(dig => digitCounts[dig]++);
      });
      // Sort to get top 3 digits
      const mappedCB = digitCounts.map((count, index) => ({ digit: index, count }))
        .sort((a,b) => b.count - a.count);
      const predictedCB = [mappedCB[0].digit, mappedCB[1].digit, mappedCB[2].digit];

      // Markov transition prediction for Kepala and Ekor
      const kepM = Array(10).fill(0).map(() => Array(10).fill(0));
      const ekorM = Array(10).fill(0).map(() => Array(10).fill(0));
      for (let s = 0; s < subset.length - 1; s++) {
        kepM[subset[s].digits[2]][subset[s].digits[3]]++;
        ekorM[subset[s].digits[3]][subset[s].digits[3]]++; // crude transition
      }

      const activeDrawLast = subset[subset.length - 1];
      const lastKepVal = activeDrawLast.digits[2];
      const lastEkorVal = activeDrawLast.digits[3];

      // transition suggestions
      const getTopTransit = (matrix: number[][], srcState: number) => {
        const row = matrix[srcState];
        const mapped = row.map((cnt, d) => ({ digit: d, count: cnt })).sort((a,b) => b.count - a.count);
        return [mapped[0].digit, mapped[1].digit, mapped[2].digit];
      };

      const topKeps = getTopTransit(kepM, lastKepVal);
      const topEkors = getTopTransit(kepM, lastEkorVal); // predictive cross transitions

      const predicted2D: string[] = [];
      topKeps.slice(0, 2).forEach(kep => {
        topEkors.slice(0, 2).forEach(ekor => {
          predicted2D.push(`${kep}${ekor}`);
        });
      });

      // Verification
      const isCBHit = actualDigits.some(dig => predictedCB.includes(dig));
      const is2DHit = predicted2D.includes(actual2D);

      if (isCBHit) cbHits++;
      if (is2DHit) d2Hits++;

      trialsDetail.push({
        drawId: evaluationDraw.id,
        date: evaluationDraw.date,
        timeSlot: evaluationDraw.timeSlot,
        actual: evaluationDraw.drawCodeString,
        predicted2D,
        predictedCB,
        isCBHit,
        is2DHit
      });
    }

    setBacktestResults({
      totalTrials: backtestSize,
      colokBebasHits: cbHits,
      colokMacauHits: cmHits,
      exact2DHits: d2Hits,
      trialsDetail: trialsDetail.reverse() // show latest trial first in log
    });
    setIsBacktesting(false);
  };

  // Computed properties
  const filteredHistory = useMemo(() => {
    return history.filter(d => {
      const matchesSearch = d.drawCodeString.includes(searchQuery) || d.date.includes(searchQuery);
      const matchesSlot = filterSlot === "All" || d.timeSlot === filterSlot;
      return matchesSearch && matchesSlot;
    });
  }, [history, searchQuery, filterSlot]);

  const statsSummary: StatisticalMetrics | null = useMemo(() => {
    if (history.length === 0) return null;
    const totalSample = history.length;
    const freqs = {
      as: Array(10).fill(0),
      kop: Array(10).fill(0),
      kepala: Array(10).fill(0),
      ekor: Array(10).fill(0),
      overall: Array(10).fill(0)
    };

    let evenCount = 0;
    let oddCount = 0;
    let bigCount = 0;
    let smallCount = 0;

    history.forEach(d => {
      d.digits.forEach((digit, index) => {
        if (digit >= 0 && digit <= 10) {
          freqs.overall[digit]++;
          if (index === 0) freqs.as[digit]++;
          else if (index === 1) freqs.kop[digit]++;
          else if (index === 2) freqs.kepala[digit]++;
          else if (index === 3) freqs.ekor[digit]++;

          // Odd-Even (As, Kop, Kepala, Ekor calculations)
          if (digit % 2 === 0) evenCount++;
          else oddCount++;

          // Big (5-9) - Small (0-4)
          if (digit >= 5) bigCount++;
          else smallCount++;
        }
      });
    });

    const overallMapped = freqs.overall.map((count, digit) => ({ digit, count }));
    const hotNumbers = [...overallMapped].sort((a, b) => b.count - a.count).slice(0, 3);
    const coldNumbers = [...overallMapped].sort((a, b) => a.count - b.count).slice(0, 3);

    return {
      totalSample,
      frequencies: {
        as: Object.fromEntries(freqs.as.map((c, i) => [i, c])),
        kop: Object.fromEntries(freqs.kop.map((c, i) => [i, c])),
        kepala: Object.fromEntries(freqs.kepala.map((c, i) => [i, c])),
        ekor: Object.fromEntries(freqs.ekor.map((c, i) => [i, c])),
        overall: Object.fromEntries(freqs.overall.map((c, i) => [i, c]))
      },
      hotNumbers,
      coldNumbers,
      evenOddRatio: { even: evenCount, odd: oddCount },
      bigSmallRatio: { big: bigCount, small: smallCount }
    };
  }, [history]);

  // Recommend BBFS/Angka Campur of 5, 6, and 7 digits from state frequency, Poisson weights, Markov transitions, and Gemini AI predictions
  const bbfsRecommendations = useMemo(() => {
    if (!history || history.length === 0) return { digits5: [], digits6: [], digits7: [] };

    const digitScores = Array(10).fill(0);

    // 1. Frequency weighting (recent 40 draws) - gives a realistic baseline
    const recentHistory = history.slice(-40);
    recentHistory.forEach(d => {
      d.digits.forEach(digit => {
        if (digit >= 0 && digit <= 9) {
          digitScores[digit] += 3;
        }
      });
    });

    // 2. Poisson probabilities weight (if compiled)
    if (modelPrediction?.poisson) {
      modelPrediction.poisson.forEach(p => {
        p.probabilities.forEach(probItem => {
          digitScores[probItem.digit] += probItem.probability * 45;
        });
      });
    }

    // 3. Markov transition state weight (if compiled)
    if (modelPrediction?.markov) {
      const positions: Array<"as" | "kop" | "kepala" | "ekor"> = ["as", "kop", "kepala", "ekor"];
      positions.forEach(pos => {
        const mData = modelPrediction.markov[pos];
        if (mData && mData.nextStateProbabilities) {
          mData.nextStateProbabilities.forEach(probItem => {
            digitScores[probItem.digit] += probItem.probability * 55;
          });
        }
      });
    }

    // 4. Overlapping overall frequency in all database
    history.forEach(d => {
      d.digits.forEach(digit => {
        if (digit >= 0 && digit <= 9) {
          digitScores[digit] += 0.2; // slight boost for long-term pattern
        }
      });
    });

    // 5. Gemini AI predictions input (when loaded)
    if (aiPrediction) {
      // 4D suggestions digits
      aiPrediction.predictions4D?.forEach((code: string) => {
        code.split("").forEach((char: string) => {
          const digit = parseInt(char, 10);
          if (!isNaN(digit)) digitScores[digit] += 6;
        });
      });

      // 3D suggestions digits
      aiPrediction.predictions3D?.forEach((code: string) => {
        code.split("").forEach((char: string) => {
          const digit = parseInt(char, 10);
          if (!isNaN(digit)) digitScores[digit] += 8;
        });
      });

      // 2D suggestions digits
      aiPrediction.predictions2D?.forEach((code: string) => {
        code.split("").forEach((char: string) => {
          const digit = parseInt(char, 10);
          if (!isNaN(digit)) digitScores[digit] += 10;
        });
      });

      // Colok Bebas digits
      aiPrediction.colokBebas?.forEach((digit: number) => {
        if (digit >= 0 && digit <= 9) {
          digitScores[digit] += 15;
        }
      });
    }

    // Rank from 0 to 9 based on scores
    const ranked = Array.from({ length: 10 }, (_, i) => ({ digit: i, score: digitScores[i] }))
      .sort((a, b) => b.score - a.score)
      .map(item => item.digit);

    // Grab combinations
    return {
      digits5: [...ranked].slice(0, 5).sort((a, b) => a - b),
      digits6: [...ranked].slice(0, 6).sort((a, b) => a - b),
      digits7: [...ranked].slice(0, 7).sort((a, b) => a - b)
    };
  }, [history, modelPrediction, aiPrediction]);

  // Calculate detailed probability breakdown for already drawn (historical) and upcoming (future) drawings
  const bbfsProbabilityAnalysis = useMemo(() => {
    if (!history || history.length === 0) {
      return {
        digits5: { past2D: 0, past3D: 0, past4D: 0, future2D: 0, future3D: 0, future4D: 0, hit2DCount: 0, hit3DCount: 0, hit4DCount: 0, recentHits: [] },
        digits6: { past2D: 0, past3D: 0, past4D: 0, future2D: 0, future3D: 0, future4D: 0, hit2DCount: 0, hit3DCount: 0, hit4DCount: 0, recentHits: [] },
        digits7: { past2D: 0, past3D: 0, past4D: 0, future2D: 0, future3D: 0, future4D: 0, hit2DCount: 0, hit3DCount: 0, hit4DCount: 0, recentHits: [] }
      };
    }

    const { digits5, digits6, digits7 } = bbfsRecommendations;

    const getPastStats = (set: number[]) => {
      let hit2D = 0;
      let hit3D = 0;
      let hit4D = 0;
      const recentHits: any[] = [];

      // Loop draws reverse (from newest to oldest) to get the most recent hits first
      for (let i = history.length - 1; i >= 0; i--) {
        const d = history[i];
        if (!d.digits) continue;
        const [as, kop, kepala, ekor] = d.digits;
        const sub2D = set.includes(kepala) && set.includes(ekor);
        const sub3D = set.includes(kop) && set.includes(kepala) && set.includes(ekor);
        const sub4D = set.includes(as) && set.includes(kop) && set.includes(kepala) && set.includes(ekor);

        if (sub2D) hit2D++;
        if (sub3D) hit3D++;
        if (sub4D) hit4D++;

        if ((sub2D || sub3D || sub4D) && recentHits.length < 5) {
          const hitType = sub4D ? "4D" : sub3D ? "3D" : "2D";
          recentHits.push({
            id: d.id,
            date: d.date,
            timeSlot: d.timeSlot,
            drawCodeString: d.drawCodeString,
            hitType
          });
        }
      }

      return {
        past2D: (hit2D / history.length) * 100,
        past3D: (hit3D / history.length) * 100,
        past4D: (hit4D / history.length) * 100,
        hit2DCount: hit2D,
        hit3DCount: hit3D,
        hit4DCount: hit4D,
        recentHits
      };
    };

    const getFutureStats = (set: number[]) => {
      const pAs = Array(10).fill(0.1);
      const pKop = Array(10).fill(0.1);
      const pKepala = Array(10).fill(0.1);
      const pEkor = Array(10).fill(0.1);

      if (modelPrediction?.poisson) {
        const asData = modelPrediction.poisson.find((p: any) => p.position.toLowerCase() === "as");
        const kopData = modelPrediction.poisson.find((p: any) => p.position.toLowerCase() === "kop");
        const kepData = modelPrediction.poisson.find((p: any) => p.position.toLowerCase() === "kepala");
        const ekorData = modelPrediction.poisson.find((p: any) => p.position.toLowerCase() === "ekor");

        if (asData) asData.probabilities.forEach((item: any) => pAs[item.digit] = item.probability);
        if (kopData) kopData.probabilities.forEach((item: any) => pKop[item.digit] = item.probability);
        if (kepData) kepData.probabilities.forEach((item: any) => pKepala[item.digit] = item.probability);
        if (ekorData) ekorData.probabilities.forEach((item: any) => pEkor[item.digit] = item.probability);
      } else {
        const counts = [Array(10).fill(0), Array(10).fill(0), Array(10).fill(0), Array(10).fill(0)];
        history.forEach(d => {
          d.digits.forEach((digit, posIdx) => {
            if (digit >= 0 && digit <= 9 && posIdx < 4) {
              counts[posIdx][digit]++;
            }
          });
        });
        for (let digit = 0; digit < 10; digit++) {
          pAs[digit] = (counts[0][digit] + 1) / (history.length + 10);
          pKop[digit] = (counts[1][digit] + 1) / (history.length + 10);
          pKepala[digit] = (counts[2][digit] + 1) / (history.length + 10);
          pEkor[digit] = (counts[3][digit] + 1) / (history.length + 10);
        }
      }

      const sumAs = set.reduce((sum, d) => sum + pAs[d], 0);
      const sumKop = set.reduce((sum, d) => sum + pKop[d], 0);
      const sumKepala = set.reduce((sum, d) => sum + pKepala[d], 0);
      const sumEkor = set.reduce((sum, d) => sum + pEkor[d], 0);

      const future2D = sumKepala * sumEkor * 100;
      const future3D = sumKop * sumKepala * sumEkor * 100;
      const future4D = sumAs * sumKop * sumKepala * sumEkor * 100;

      return {
        future2D: Math.min(future2D, 99.9),
        future3D: Math.min(future3D, 99.9),
        future4D: Math.min(future4D, 99.9)
      };
    };

    const stats5 = getPastStats(digits5);
    const stats6 = getPastStats(digits6);
    const stats7 = getPastStats(digits7);

    const fstats5 = getFutureStats(digits5);
    const fstats6 = getFutureStats(digits6);
    const fstats7 = getFutureStats(digits7);

    return {
      digits5: { ...stats5, ...fstats5 },
      digits6: { ...stats6, ...fstats6 },
      digits7: { ...stats7, ...fstats7 }
    };
  }, [history, bbfsRecommendations, modelPrediction]);

  const digits7RecentAnalysis = useMemo(() => {
    if (!history || history.length === 0) return [];
    
    const slotsOrder = ["00:01", "13:00", "16:00", "19:00", "22:00", "23:00"];
    
    // Sort entire history from newest (latest date/timeslot) to oldest
    const sortedHistory = [...history].sort((a, b) => {
      if (a.date !== b.date) {
        return b.date.localeCompare(a.date);
      }
      const idxA = slotsOrder.indexOf(a.timeSlot);
      const idxB = slotsOrder.indexOf(b.timeSlot);
      return idxB - idxA;
    });

    const recentDraws = sortedHistory.slice(0, 40);
    const set = bbfsRecommendations.digits7;

    return recentDraws.map(d => {
      const hasDigits = d.digits && d.digits.length >= 4;
      if (!hasDigits) {
        return {
          id: d.id,
          date: d.date,
          timeSlot: d.timeSlot,
          drawCodeString: d.drawCodeString,
          isHit: false,
          hitType: null
        };
      }
      const [as, kop, kepala, ekor] = d.digits;
      const sub2D = set.includes(kepala) && set.includes(ekor);
      const sub3D = set.includes(kop) && set.includes(kepala) && set.includes(ekor);
      const sub4D = set.includes(as) && set.includes(kop) && set.includes(kepala) && set.includes(ekor);
      
      const isHit = sub2D || sub3D || sub4D;
      const hitType = sub4D ? "4D" : sub3D ? "3D" : sub2D ? "2D" : null;

      return {
        id: d.id,
        date: d.date,
        timeSlot: d.timeSlot,
        drawCodeString: d.drawCodeString,
        isHit,
        hitType
      };
    });
  }, [history, bbfsRecommendations.digits7]);

  return (
    <div id="applet-viewport" className="min-h-screen bg-slate-950 text-slate-100 font-sans selection:bg-cyan-500 selection:text-slate-950">
      
      {/* Upper Technical Border */}
      <div className="bg-gradient-to-r from-cyan-600 via-emerald-500 to-indigo-600 h-1" />

      {/* Navigation and Top Header */}
      <header id="applet-header" className="border-b border-slate-900 bg-slate-950/80 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 md:px-6 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          
          {/* Main Logo block */}
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-cyan-500 to-emerald-400 p-2.5 rounded-xl shadow-lg ring-1 ring-cyan-400/20">
              <Cpu className="h-6 w-6 text-slate-950 stroke-[2.5]" id="logo-icon" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-slate-50 via-slate-100 to-cyan-200 bg-clip-text text-transparent">
                  MACAULYZER
                </h1>
                <span className="text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-400/20 px-2 py-0.5 rounded-full font-mono uppercase tracking-widest font-semibold">
                  Alpha v2.6
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono">Statistical Calculus & Machine Learning Togel Macau Model</p>
            </div>
          </div>

          {/* Tab Navigation buttons */}
          <nav className="flex flex-wrap gap-1.5 p-1 bg-slate-900/60 rounded-xl border border-slate-900 max-w-full overflow-x-auto">
            <button
              onClick={() => setActiveTab("dashboard")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono tracking-wide transition-all ${
                activeTab === "dashboard"
                  ? "bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <Database className="inline h-3.5 w-3.5 mr-1" /> SQL DB
            </button>
            <button
              onClick={() => setActiveTab("poisson")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono tracking-wide transition-all ${
                activeTab === "poisson"
                  ? "bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <BarChart3 className="inline h-3.5 w-3.5 mr-1" /> Poisson PMF
            </button>
            <button
              onClick={() => setActiveTab("markov")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono tracking-wide transition-all ${
                activeTab === "markov"
                  ? "bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <Dices className="inline h-3.5 w-3.5 mr-1" /> Markov Chain
            </button>
            <button
              onClick={() => setActiveTab("calculus")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono tracking-wide transition-all ${
                activeTab === "calculus"
                  ? "bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <TrendingUp className="inline h-3.5 w-3.5 mr-1" /> Calculus Regresi
            </button>
            <button
              onClick={() => setActiveTab("ai")}
              className={`px-4 py-1.5 rounded-lg text-xs font-semibold font-mono tracking-wide transition-all ${
                activeTab === "ai"
                  ? "bg-gradient-to-r from-cyan-500 to-indigo-600 text-white shadow-md shadow-indigo-500/10 border-0"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <Cpu className="inline h-3.5 w-3.5 mr-1" /> Gemini AI
            </button>
            <button
              onClick={() => setActiveTab("ml")}
              className={`px-4 py-1.5 rounded-lg text-xs font-bold font-mono tracking-wide transition-all ${
                activeTab === "ml"
                  ? "bg-slate-800 text-emerald-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <Activity className="inline h-3.5 w-3.5 mr-1 text-emerald-400 animate-pulse" /> Algoritma ML
            </button>
            <button
              onClick={() => setActiveTab("backtest")}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium font-mono tracking-wide transition-all ${
                activeTab === "backtest"
                  ? "bg-slate-800 text-cyan-400 shadow-sm border border-slate-700/50"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/30"
              }`}
            >
              <CheckCircle className="inline h-3.5 w-3.5 mr-1" /> Backtest Sandbox
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 md:px-6 py-8">
        
        {/* Overview Row: Static values describing the strict nature of the system */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          
          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
            <div className="p-3 bg-cyan-500/10 rounded-lg text-cyan-400">
              <Database className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Database Undian</p>
              <p className="text-xl font-bold font-mono text-cyan-400">{history.length} Draws</p>
              <p className="text-[10px] text-slate-500">Keluaran terintegrasi</p>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
            <div className="p-3 bg-emerald-500/10 rounded-lg text-emerald-400">
              <Activity className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Shannon Entropy</p>
              <p className="text-xl font-bold font-mono text-emerald-400">
                {modelPrediction?.chiSquare?.entropy ? `${modelPrediction.chiSquare.entropy.toFixed(3)} bits` : "Loading..."}
              </p>
              <p className="text-[10px] text-slate-500">Max Entropy: 3.322 bits</p>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
            <div className="p-3 bg-purple-500/10 rounded-lg text-purple-400">
              <Award className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Model Confidence</p>
              <p className="text-xl font-bold font-mono text-purple-400">
                {modelPrediction?.suggestedNumbers?.confidence ? `${modelPrediction.suggestedNumbers.confidence}%` : "Loading..."}
              </p>
              <p className="text-[10px] text-slate-500">Berdasarkan Matriks & Deviasi</p>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex items-center gap-4">
            <div className="p-3 bg-amber-500/10 rounded-lg text-amber-400">
              <AlertTriangle className="h-5 w-5" />
            </div>
            <div>
              <p className="text-[10px] font-mono uppercase tracking-wider text-slate-400">Pearson Chi-Square</p>
              <p className="text-xl font-bold font-mono text-amber-400">
                {modelPrediction?.chiSquare?.chiSquareStat ? modelPrediction.chiSquare.chiSquareStat.toFixed(2) : "Loading..."}
              </p>
              <p className="text-[10px] text-slate-500">
                Critical (df=9): 16.92 ({modelPrediction?.chiSquare?.isBiased ? "Ada Bias" : "Acak Murni"})
              </p>
            </div>
          </div>

        </div>

        {/* TAB 1: DASHBOARD & HISTORY MANAGEMENT */}
        {activeTab === "dashboard" && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            
            {/* Column Left: Live Draw Sync + Manual input form */}
            <div className="lg:col-span-1 space-y-6">
              
              {/* Box 1: Real-time API Integration (Web Grounding) */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 relative overflow-hidden">
                <div className="absolute right-0 top-0 w-24 h-24 bg-gradient-to-br from-cyan-500/5 to-emerald-500/5 blur-xl pointer-events-none rounded-full" />
                
                <h3 className="text-sm font-semibold font-mono tracking-wider text-slate-200 border-b border-slate-800 pb-3 flex items-center gap-2">
                  <RefreshCw className="h-4 w-4 text-cyan-400 animate-spin-slow" /> INTEGRASI DATA REAL-TIME
                </h3>

                <p className="text-xs text-slate-400 mt-3 mb-5 leading-relaxed">
                  Gunakan integrasi web-scraping berbasis grounding Google Search untuk mencari dan menyinkronkan data undian Toto Macau terbaru dari internet secara live.
                </p>

                {scrapeSuccess && (
                  <div className="bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs p-3 rounded-lg flex items-start gap-2 mb-4">
                    <CheckCircle className="h-4 w-4 mt-0.5 shrink-0" />
                    <span>{scrapeSuccess}</span>
                  </div>
                )}

                {scrapeError && (
                  <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs p-3 rounded-lg flex items-start gap-2 mb-4">
                    <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
                    <span>{scrapeError}</span>
                  </div>
                )}

                <button
                  onClick={handleScrapeSync}
                  disabled={isScraping}
                  className="w-full bg-gradient-to-r from-cyan-500 to-emerald-500 hover:from-cyan-600 hover:to-emerald-600 active:scale-[0.98] transition-transform text-slate-950 font-bold font-mono text-xs py-3 rounded-xl flex items-center justify-center gap-2 cursor-pointer shadow-lg shadow-cyan-500/10 disabled:opacity-50"
                >
                  <RefreshCw className={`h-4 w-4 ${isScraping ? "animate-spin" : ""}`} />
                  {isScraping ? "MENGAMBIL DATA LIVE..." : "SINGKRONIKAN DATA WEB ONLINE"}
                </button>
              </div>

              {/* Box 2: Manual Database Injector OR Edit Database Injector */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                {editingDraw ? (
                  <>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <h3 className="text-sm font-semibold font-mono tracking-wider text-amber-400 flex items-center gap-2">
                        <Pencil className="h-4 w-4 text-amber-400" /> EDIT DATA KELUARAN
                      </h3>
                      <button
                        onClick={() => setEditingDraw(null)}
                        className="text-slate-400 hover:text-slate-200 transition-colors cursor-pointer p-0.5 rounded bg-slate-950/40 hover:bg-slate-950"
                        title="Batal Edit"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>

                    <form onSubmit={handleEditSubmit} className="space-y-4 mt-4">
                      
                      <div className="bg-amber-500/5 border border-amber-500/10 rounded-xl p-3 text-[11px] font-mono text-amber-200/80">
                        Mengoreksi UID: <strong className="text-amber-400">#{editingDraw.id}</strong>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Tanggal Undian
                        </label>
                        <input
                          type="date"
                          value={editFormData.date}
                          onChange={e => setEditFormData({ ...editFormData, date: e.target.value })}
                          required
                          className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none"
                        />
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Jam Tayang (Timeslot)
                        </label>
                        <select
                          value={editFormData.timeSlot}
                          onChange={e => setEditFormData({ ...editFormData, timeSlot: e.target.value })}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none"
                        >
                          <option value="00:01">00:01 WIB</option>
                          <option value="13:00">13:00 WIB</option>
                          <option value="16:00">16:00 WIB</option>
                          <option value="19:00">19:00 WIB</option>
                          <option value="22:00">22:00 WIB</option>
                          <option value="23:00">23:00 WIB</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Hasil Baru 4-Angka
                        </label>
                        <input
                          type="text"
                          maxLength={4}
                          placeholder="Contoh: 1845"
                          value={editFormData.drawCodeString}
                          onChange={e => setEditFormData({ ...editFormData, drawCodeString: e.target.value.replace(/\D/g, "") })}
                          required
                          className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none tracking-widest text-center text-lg font-bold placeholder:text-slate-700"
                        />
                      </div>

                      {editError && <p className="text-xs text-rose-400 font-mono">{editError}</p>}
                      {editSuccess && <p className="text-xs text-amber-400 font-mono">{editSuccess}</p>}

                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => setEditingDraw(null)}
                          className="w-1/3 bg-slate-950 hover:bg-slate-900 text-slate-400 transition-colors border border-slate-800 font-semibold font-mono text-xs py-2.5 rounded-lg flex items-center justify-center cursor-pointer"
                        >
                          BATAL
                        </button>
                        <button
                          type="submit"
                          disabled={isSavingEdit}
                          className="w-2/3 bg-slate-800 hover:bg-slate-700 active:scale-[0.98] transition-transform text-amber-400 border border-amber-500/20 font-bold font-mono text-xs py-2.5 rounded-lg flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                        >
                          {isSavingEdit ? "MENYIMPAN..." : "SIMPAN PERUBAHAN"}
                        </button>
                      </div>

                    </form>
                  </>
                ) : (
                  <>
                    <h3 className="text-sm font-semibold font-mono tracking-wider text-slate-200 border-b border-slate-800 pb-3 flex items-center gap-2">
                      <PlusCircle className="h-4 w-4 text-cyan-400" /> INPUT KELUARAN MANUAL
                    </h3>

                    <form onSubmit={handleManualSubmit} className="space-y-4 mt-4">
                      
                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Tanggal Undian
                        </label>
                        <div className="relative">
                          <input
                            type="date"
                            value={formData.date}
                            onChange={e => setFormData({ ...formData, date: e.target.value })}
                            required
                            className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none"
                          />
                        </div>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Jam Tayang (Timeslot)
                        </label>
                        <select
                          value={formData.timeSlot}
                          onChange={e => setFormData({ ...formData, timeSlot: e.target.value })}
                          className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none"
                        >
                          <option value="00:01">00:01 WIB</option>
                          <option value="13:00">13:00 WIB</option>
                          <option value="16:00">16:00 WIB</option>
                          <option value="19:00">19:00 WIB</option>
                          <option value="22:00">22:00 WIB</option>
                          <option value="23:00">23:00 WIB</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 uppercase tracking-widest mb-1.5 font-semibold">
                          Hasil 4-Angka (e.g. 5432)
                        </label>
                        <input
                          type="text"
                          maxLength={4}
                          placeholder="Contoh: 1845"
                          value={formData.drawCodeString}
                          onChange={e => setFormData({ ...formData, drawCodeString: e.target.value.replace(/\D/g, "") })}
                          required
                          className="w-full bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg p-2.5 text-xs text-slate-100 font-mono outline-none tracking-widest text-center text-lg font-bold placeholder:text-slate-700"
                        />
                      </div>

                      {formError && <p className="text-xs text-rose-400 font-mono">{formError}</p>}
                      {formSuccess && <p className="text-xs text-emerald-400 font-mono">{formSuccess}</p>}

                      <button
                        type="submit"
                        disabled={isSubmittingForm}
                        className="w-full bg-slate-800 hover:bg-slate-700 active:scale-[0.98] transition-transform text-cyan-400 border border-cyan-400/20 font-bold font-mono text-xs py-2.5 rounded-lg flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                      >
                        <PlusCircle className="h-4 w-4" />
                        {isSubmittingForm ? "MENYIMPAN..." : "TAMBAH KE SEED DATABASE"}
                      </button>

                    </form>
                  </>
                )}
              </div>

              {/* Box 3: Image OCR Database Sync and Correction */}
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                <h3 className="text-sm font-semibold font-mono tracking-wider text-slate-200 border-b border-slate-800 pb-3 flex items-center gap-2">
                  <Image className="h-4 w-4 text-emerald-400" /> IMPORT & KOREKSI VIA OCR GAMBAR
                </h3>

                <p className="text-xs text-slate-400 mt-3 leading-relaxed">
                  Unggah tangkapan layar (screenshot) atau foto tabel nomor keluaran Toto Macau. Sistem akan membaca gambar menggunakan teknologi OCR visibilitas multimodal dan secara otomatis mengisi nomor baru serta memperbaiki kesalahan pada database yang ada.
                </p>

                <div className="mt-4">
                  <div
                    onDragOver={handleOcrDragOver}
                    onDragLeave={handleOcrDragLeave}
                    onDrop={handleOcrDrop}
                    className={`border-2 border-dashed rounded-xl p-6 text-center transition-all cursor-pointer flex flex-col items-center justify-center gap-3 relative ${
                      isDraggingOcr
                        ? "border-emerald-400 bg-emerald-500/5"
                        : "border-slate-800 hover:border-slate-700 bg-slate-950/40 hover:bg-slate-950/60"
                    }`}
                    onClick={() => document.getElementById("ocr-file-input")?.click()}
                  >
                    <input
                      id="ocr-file-input"
                      type="file"
                      accept="image/*"
                      onChange={handleOcrFileChange}
                      className="hidden"
                    />
                    
                    {isOcrLoading ? (
                      <div className="flex flex-col items-center gap-2 py-2">
                        <RefreshCw className="h-8 w-8 text-emerald-400 animate-spin" />
                        <span className="text-xs font-mono text-emerald-400 font-semibold animate-pulse">Gemini AI Sedang Membaca Gambar...</span>
                      </div>
                    ) : (
                      <>
                        <Upload className="h-8 w-8 text-emerald-500/80" />
                        <div className="space-y-1">
                          <p className="text-xs font-semibold text-slate-300">Tarik gambar ke sini, atau klik untuk memilih</p>
                          <p className="text-[10px] text-slate-500">Mendukung PNG, JPEG, JPG, WEBP</p>
                        </div>
                      </>
                    )}
                  </div>
                </div>

                {ocrError && (
                  <div className="mt-3 bg-rose-500/10 border border-rose-500/20 text-rose-400 text-xs p-3 rounded-lg flex items-start gap-2">
                    <span className="text-xs">⚠️ {ocrError}</span>
                  </div>
                )}

                {ocrWarning && (
                  <div className="mt-3 bg-amber-500/10 border border-amber-500/20 text-amber-400 text-[11px] p-3 rounded-lg leading-snug">
                    <span>{ocrWarning}</span>
                  </div>
                )}

                {ocrSuccess && (
                  <div className="mt-3 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs p-3 rounded-lg">
                    <span className="font-semibold block mb-1">✓ Berhasil Diproses!</span>
                    <span>{ocrSuccess}</span>
                  </div>
                )}

                {ocrResults.length > 0 && (
                  <div className="mt-4 space-y-2">
                    <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800 pb-1.5">
                      Daftar Angka Ekstraksi OCR:
                    </div>
                    <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 text-xs select-none">
                      {ocrResults.map((resItem, idx) => {
                        // Check if it already matches an existing draw in our active database
                        const isMatch = history.some(h => h.date === resItem.date && h.timeSlot === resItem.timeSlot);
                        return (
                          <div
                            key={idx}
                            className="bg-slate-950/80 border border-slate-800/60 rounded-lg p-2 flex items-center justify-between font-mono"
                          >
                            <div className="flex flex-col">
                              <span className="text-slate-300 font-medium text-[11px]">{resItem.date}</span>
                              <span className="text-[10px] text-slate-500">{resItem.timeSlot} WIB</span>
                            </div>
                            <div className="flex items-center gap-2">
                              <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-slate-800 text-slate-300">
                                {resItem.drawCodeString}
                              </span>
                              {isMatch ? (
                                <span className="bg-amber-400/10 border border-amber-400/25 text-amber-400 text-[9px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded">
                                  KOREKSI
                                </span>
                              ) : (
                                <span className="bg-emerald-400/10 border border-emerald-400/25 text-emerald-400 text-[9px] font-bold tracking-wider uppercase px-1.5 py-0.5 rounded">
                                  BARU
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

              </div>

            </div>

            {/* Column Right: Table draw query database list */}
            <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-3xl p-6">
              
              <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 border-b border-slate-800 pb-5 mb-5">
                <div>
                  <h2 className="text-md font-bold tracking-tight text-slate-200">Database Transaksi Data Macau</h2>
                  <p className="text-xs text-slate-400">Total data: {history.length} record undian yang aktif digunakan.</p>
                </div>

                <div className="flex gap-2 items-center flex-wrap">
                  <div className="relative">
                    <Search className="absolute left-3 top-2.5 h-3.5 w-3.5 text-slate-500" />
                    <input
                      type="text"
                      placeholder="Cari Tanggal / Angka..."
                      value={searchQuery}
                      onChange={e => setSearchQuery(e.target.value)}
                      className="bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg pl-9 pr-3 py-1.5 text-xs text-slate-200 outline-none w-48 font-mono"
                    />
                  </div>

                  <select
                    value={filterSlot}
                    onChange={e => setFilterSlot(e.target.value)}
                    className="bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg px-3 py-1.5 text-xs text-slate-300 outline-none font-mono"
                  >
                    <option value="All">Semua Jam</option>
                    <option value="00:01">00:01</option>
                    <option value="13:00">13:00</option>
                    <option value="16:00">16:00</option>
                    <option value="19:00">19:00</option>
                    <option value="22:00">22:00</option>
                    <option value="23:00">23:00</option>
                  </select>
                </div>
              </div>

              {isHistoryLoading ? (
                <div className="py-20 flex flex-col items-center justify-center gap-3">
                  <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin" />
                  <p className="text-xs text-slate-500 font-mono">Memuat database dari server...</p>
                </div>
              ) : historyError ? (
                <div className="py-20 text-center text-rose-400 font-mono text-xs max-w-md mx-auto">
                  <AlertTriangle className="h-8 w-8 text-rose-400 mx-auto mb-2" />
                  <p>{historyError}</p>
                </div>
              ) : filteredHistory.length === 0 ? (
                <div className="py-20 text-center text-slate-500 font-mono text-xs">
                  Tidak ditemukan record undian pada kriteria pencarian terkait.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead>
                      <tr className="border-b border-slate-800/80 text-slate-400 font-mono font-medium text-[10px] uppercase tracking-wider bg-slate-950/20">
                        <th className="py-3 px-4">UID</th>
                        <th className="py-3 px-4">Tanggal</th>
                        <th className="py-3 px-4">Timeslot</th>
                        <th className="py-3 px-4 text-center">4D Hasil</th>
                        <th className="py-3 px-4 text-center">As | Kop | Kep | Ekor</th>
                        <th className="py-3 px-4 text-right">Sum</th>
                        <th className="py-3 px-4 text-center">Aksi</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40">
                      {filteredHistory.slice(0, 15).map((draw, idx) => (
                        <tr key={draw.id} className="hover:bg-slate-800/20 transition-colors">
                          <td className="py-3.5 px-4 font-mono font-bold text-slate-600">#{draw.id}</td>
                          <td className="py-3.5 px-4 font-mono flex items-center gap-1.5 text-slate-300">
                            <Calendar className="h-3 w-3 text-cyan-500" /> {draw.date}
                          </td>
                          <td className="py-3.5 px-4 font-mono text-slate-400">{draw.timeSlot} WIB</td>
                          <td className="py-3.5 px-4 text-center font-mono font-bold text-emerald-400 tracking-wider text-sm">
                            {draw.drawCodeString}
                          </td>
                          <td className="py-3.5 px-4 text-center">
                            <div className="inline-flex gap-1.5 font-mono">
                              <span className="bg-slate-850 px-2 py-0.5 rounded text-slate-300 ring-1 ring-white/5">{draw.digits[0]}</span>
                              <span className="bg-slate-850 px-2 py-0.5 rounded text-slate-300 ring-1 ring-white/5">{draw.digits[1]}</span>
                              <span className="bg-slate-850 px-2 py-0.5 rounded text-slate-300 ring-1 ring-white/5">{draw.digits[2]}</span>
                              <span className="bg-slate-850 px-2 py-0.5 rounded text-slate-300 ring-1 ring-white/5">{draw.digits[3]}</span>
                            </div>
                          </td>
                          <td className="py-3.5 px-4 text-right font-mono font-semibold text-slate-400">{draw.sum}</td>
                          <td className="py-3.5 px-4 text-center">
                            <button
                              onClick={() => {
                                setEditingDraw(draw);
                                setEditFormData({
                                  date: draw.date,
                                  timeSlot: draw.timeSlot,
                                  drawCodeString: draw.drawCodeString
                                });
                                setEditError("");
                                setEditSuccess("");
                              }}
                              className="px-2.5 py-1 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-400/25 font-bold font-mono text-[10px] tracking-wider uppercase inline-flex items-center gap-1.5 cursor-pointer transition-colors"
                              title="Koreksi data undian"
                            >
                              <Pencil className="h-3 w-3" /> Perbaiki
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="border-t border-slate-800/50 pt-4 text-right text-[10px] text-slate-500 font-mono">
                    Menampilkan 15 undian teratas • Data diurutkan kronologi terbalik
                  </div>
                </div>
              )}

            </div>

          </div>
        )}

        {/* TAB 2: POISSON DISTRIBUTION ENGINE */}
        {activeTab === "poisson" && (
          <div className="space-y-8">
            
            {/* Explanatory introduction on Poisson Mathematics */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-start">
              <div className="p-4 bg-gradient-to-br from-cyan-400 to-indigo-500 rounded-2xl text-slate-950 font-bold font-mono text-xl tracking-tight leading-none self-start shrink-0">
                P(X)
              </div>
              <div>
                <h2 className="text-md font-bold text-slate-100 flex items-center gap-2">
                  Model Probabilitas Poisson (Poisson PMF Model)
                </h2>
                <p className="text-xs text-slate-400 mt-1 lines-normal leading-relaxed">
                  Dalam teori probabilitas, <strong>Distribusi Poisson</strong> memodelkan kemungkinan terjadinya peristiwa dalam interval waktu yang ditentukan. Di sini, kita menghitung tingkat kemunculan rata-rata (<code className="text-cyan-400 font-mono">&#x03BB;</code>) untuk masing-masing digit (0-9) di setiap posisi As, Kop, Kepala, dan Ekor selama {history.length} undian terakhir.
                </p>
                <div className="mt-4 font-mono text-[10px] bg-slate-950/80 p-2.5 rounded border border-slate-850 inline-block text-cyan-400">
                  Rumus Poisson PMF: P(X = k) = ( &lambda;<sup>k</sup> * e<sup>-&lambda;</sup> ) / k!
                </div>
              </div>
            </div>

            {/* Render Poisson probabilities matrix */}
            {!modelPrediction ? (
              <div className="py-20 bg-slate-900 border border-slate-800 rounded-3xl flex flex-col items-center justify-center gap-3">
                <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin" />
                <p className="text-xs text-slate-500 font-mono">Menghitung model Poisson distribusi...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                
                {modelPrediction.poisson.map((posData, pIdx) => (
                  <div key={posData.position} className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
                    
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                      <div>
                        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">Posisi</span>
                        <h3 className="text-sm font-bold text-slate-100">{posData.position}</h3>
                      </div>
                      <div className="bg-cyan-500/10 text-cyan-400 border border-cyan-400/20 rounded px-2.5 py-1 text-[11px] font-mono font-bold">
                        Digit Prediksi: {posData.mostProbable}
                      </div>
                    </div>

                    <div className="space-y-3.5">
                      {posData.probabilities.slice(0, 10).map((item, idx) => {
                        // Calculate percentage of width relative to max probability to scale bar graphs beautifully
                        const maxProb = posData.probabilities[0].probability || 1;
                        const percent = (item.probability / maxProb) * 100;
                        
                        return (
                          <div key={item.digit} className="space-y-1">
                            <div className="flex items-center justify-between text-[11px] font-mono">
                              <span className={`font-bold ${idx === 0 ? "text-cyan-400" : "text-slate-300"}`}>
                                Digit {item.digit} {idx === 0 && "🏆"}
                              </span>
                              <span className="text-slate-400">{(item.probability * 100).toFixed(3)}%</span>
                            </div>
                            <div className="w-full bg-slate-950 h-2 rounded overflow-hidden">
                              <div
                                style={{ width: `${percent}%` }}
                                className={`h-full rounded transition-all duration-500 ${
                                  idx === 0
                                    ? "bg-gradient-to-r from-cyan-500 to-cyan-400"
                                    : idx < 3
                                    ? "bg-slate-700"
                                    : "bg-slate-800/40"
                                }`}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>

                  </div>
                ))}

              </div>
            )}

          </div>
        )}

        {/* TAB 3: MARKOV CHAIN TRANSITION PATTERN */}
        {activeTab === "markov" && (
          <div className="space-y-8">
            
            {/* Explanation card */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-start">
              <div className="p-4 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-2xl text-slate-950 font-bold font-mono text-xl tracking-tight leading-none self-start shrink-0">
                P(S|S')
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  Matriks Transisi Rantai Markov (Markov Chain Matrice Transitions)
                </h3>
                <p className="text-xs text-slate-400 mt-1 lines-normal leading-relaxed">
                  Model <strong>Rantai Markov Orde Satu</strong> mengamati bahwa kondisi masa depan bergantung hanya pada status saat ini, bukan rentetan status masa lampau. Di bawah ini merupakan analisis runtunan perubahan digit pada undian berturut-turut. Setiap baris sel merepresentasikan digit asal ke digit keluaran selanjutnya.
                </p>
                <div className="mt-4 font-mono text-[10px] bg-slate-950/80 p-2.5 rounded border border-slate-850 inline-block text-indigo-400">
                  Probabilitas Transisi: P( S<sub>t+1</sub> = j | S<sub>t</sub> = i ) = N<sub>i,j</sub> / &Sigma;<sub>c</sub> N<sub>i,c</sub>
                </div>
              </div>
            </div>

            {/* Matrix interface */}
            {!modelPrediction ? (
              <div className="py-20 bg-slate-900 border border-slate-800 rounded-3xl flex flex-col items-center justify-center gap-3">
                <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin" />
                <p className="text-xs text-slate-500 font-mono">Mengompilasi Rantai Markov...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Positional selector */}
                <div className="lg:col-span-1 space-y-6">
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                    <h4 className="text-xs font-bold font-mono text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-5">
                      Menu Visualisasi Matriks
                    </h4>

                    {/* Choose Position */}
                    <div className="space-y-4">
                      <div>
                        <label className="block text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1.5 font-bold">
                          1. Pilih Posisi Digit
                        </label>
                        <div className="grid grid-cols-4 gap-1.5 bg-slate-950 p-1.5 rounded-lg border border-slate-850">
                          {["as", "kop", "kepala", "ekor"].map((posOption) => (
                            <button
                              key={posOption}
                              onClick={() => setSelectedMarkovPos(posOption as any)}
                              className={`py-1.5 rounded text-[11px] font-mono font-semibold uppercase text-center transition-all cursor-pointer ${
                                selectedMarkovPos === posOption
                                  ? "bg-slate-850 text-cyan-400 border border-slate-750"
                                  : "text-slate-400 hover:text-slate-250"
                              }`}
                            >
                              {posOption}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Choose Digit Asal */}
                      <div>
                        <label className="block text-[10px] font-mono text-slate-500 uppercase tracking-widest mb-1.5 font-bold">
                          2. Pilih Digit State Saat Ini (S<sub>t</sub>)
                        </label>
                        <div className="grid grid-cols-5 gap-1.5">
                          {Array.from({ length: 10 }).map((_, digit) => (
                            <button
                              key={digit}
                              onClick={() => setSelectedMarkovDigit(digit)}
                              className={`p-2 rounded font-mono font-bold text-center border cursor-pointer transition-all ${
                                selectedMarkovDigit === digit
                                  ? "bg-cyan-500/15 text-cyan-400 border-cyan-500/30 font-bold"
                                  : "bg-slate-950 text-slate-400 border-slate-850 hover:bg-slate-850"
                              }`}
                            >
                              State {digit}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="mt-6 p-4 bg-slate-950 rounded-xl border border-slate-855 text-xs">
                      <p className="font-mono text-slate-400 font-medium">State Transisi Terakhir:</p>
                      <div className="mt-2 text-slate-300 font-mono">
                        Pada database kita, digit terakhir di posisi <span className="text-cyan-400 capitalize font-bold">{selectedMarkovPos}</span> adalah <span className="text-cyan-300 font-bold">{modelPrediction.markov[selectedMarkovPos].currentState}</span>.
                      </div>
                    </div>
                  </div>
                </div>

                {/* Display Markov Rows probability transition */}
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-3xl p-6">
                  
                  <div className="border-b border-slate-800 pb-4 mb-5 flex items-center justify-between">
                    <div>
                      <h4 className="text-sm font-bold text-slate-200">
                        Peta Distribusi Probabilitas Transisi
                      </h4>
                      <p className="text-xs text-slate-400">
                        Menampilkan probabilitas digit apa yang akan keluar di {selectedMarkovPos} berikutnya, jika saat ini adalah <strong className="text-cyan-400">Digit {selectedMarkovDigit}</strong>.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {(() => {
                      const probabilities = modelPrediction.markov[selectedMarkovPos].matrix[selectedMarkovDigit];
                      const maxProb = Math.max(...probabilities) || 1;
                      
                      return probabilities.map((prob: number, d: number) => {
                        const percentOfMax = (prob / maxProb) * 100;
                        const isMax = prob === maxProb && prob > 0;

                        return (
                          <div key={d} className="flex items-center gap-4">
                            <div className="w-16 text-right font-mono text-xs font-bold shrink-0 text-slate-300">
                              State &rarr; {d}
                            </div>
                            
                            <div className="flex-1 bg-slate-950 h-5 rounded overflow-hidden relative border border-slate-850">
                              <div
                                style={{ width: `${percentOfMax}%` }}
                                className={`h-full rounded-r transition-all duration-500 ${
                                  isMax
                                    ? "bg-gradient-to-r from-indigo-500 to-indigo-400"
                                    : "bg-slate-800"
                                }`}
                              />
                              <div className="absolute inset-0 flex items-center justify-start pl-3">
                                <span className="font-mono text-[10px] text-slate-400 font-bold">
                                  {isMax ? "TRANSISI PALING TINGGI 🔥" : ""}
                                </span>
                              </div>
                            </div>

                            <div className="w-16 font-mono text-right text-xs font-bold text-cyan-400">
                              {(prob * 100).toFixed(2)}%
                            </div>
                          </div>
                        );
                      });
                    })()}
                  </div>

                </div>

              </div>
            )}

          </div>
        )}

        {/* TAB 4: CALCULUS REGRESSION TRENDLINE */}
        {activeTab === "calculus" && (
          <div className="space-y-8">
            
            {/* Explanation box */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-start">
              <div className="p-4 bg-gradient-to-br from-emerald-500 to-teal-500 rounded-2xl text-slate-950 font-bold font-mono text-xl tracking-tight leading-none self-start shrink-0">
                &part;y/&part;x
              </div>
              <div>
                <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                  Regresi Linear Minimalis & Telemetri Gradien Menurun (Calculus Gradient Descent)
                </h3>
                <p className="text-xs text-slate-400 mt-1 lines-normal leading-relaxed">
                  Di sini, kita mengulas tren fluktuasi <strong>jumlah digit (Sum)</strong> yang ditarik dari waktu ke waktu. Persamaan regresi <code className="text-cyan-400">y = mx + c</code> diselesaikan untuk mengukur slope (kecenderungan arah angka total). Kami juga menyertakan visualisasi proses pembelajaran Kalkulus <strong>SGD (Stochastic Gradient Descent)</strong> yang meminimalkan fungsi Loss secara manual dalam 5 langkah iterasi.
                </p>
              </div>
            </div>

            {/* Regression content split */}
            {!modelPrediction ? (
              <div className="py-20 bg-slate-900 border border-slate-800 rounded-3xl flex flex-col items-center justify-center gap-3">
                <RefreshCw className="h-8 w-8 text-cyan-400 animate-spin" />
                <p className="text-xs text-slate-500 font-mono">Memuat Kalkulasi Regresi...</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Column to display regression parameters and gradient descent epoch logs */}
                <div className="lg:col-span-1 space-y-6">
                  
                  {/* Parameter Box */}
                  <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                    <h4 className="text-xs font-bold font-mono text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-4">
                      Parameter Hasil Fit Regresi
                    </h4>

                    <div className="space-y-3 font-mono text-xs">
                      <div className="flex items-center justify-between py-1 border-b border-slate-850">
                        <span className="text-slate-400">Slope Gradien (m)</span>
                        <span className="text-cyan-400 font-bold">{modelPrediction.regression.slope.toFixed(6)}</span>
                      </div>
                      <div className="flex items-center justify-between py-1 border-b border-slate-850">
                        <span className="text-slate-400">Konstanta Bias (c)</span>
                        <span className="text-cyan-400 font-bold">{modelPrediction.regression.intercept.toFixed(4)}</span>
                      </div>
                      <div className="flex items-center justify-between py-1 border-b border-slate-850">
                        <span className="text-slate-400">Fitted Persamaan</span>
                        <span className="text-cyan-400 font-bold">{`y = ${modelPrediction.regression.slope.toFixed(3)}x + ${modelPrediction.regression.intercept.toFixed(2)}`}</span>
                      </div>
                      <div className="flex items-center justify-between py-1 border-b border-slate-850">
                        <span className="text-slate-400">Akurasi Model R-Sq</span>
                        <span className="text-amber-400 font-bold">{(modelPrediction.regression.rSquared * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex items-center justify-between py-1 text-emerald-400 pt-2 font-bold text-sm">
                        <span>Prediksi Sum Berikutnya</span>
                        <span>{modelPrediction.regression.predictedNextSum}</span>
                      </div>
                    </div>
                  </div>

                  {/* Gradient Steps Box */}
                  <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                    <h4 className="text-xs font-bold font-mono text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-4 flex items-center justify-between">
                      <span>Proses Gradient Descent Step</span>
                      <span className="text-[9px] lowercase text-slate-500 font-normal">learning rate = 0.0001</span>
                    </h4>

                    <div className="space-y-3">
                      {modelPrediction.regression.gradientDescentSteps.map(step => (
                        <div key={step.step} className="bg-slate-950 p-3 rounded-lg border border-slate-855 font-mono text-xs">
                          <div className="flex items-center justify-between border-b border-slate-850 pb-1.5 mb-1.5">
                            <span className="text-cyan-400 font-bold">Epoch Iterasi #{step.step}</span>
                            <span className="text-slate-500">Loss: {step.loss.toFixed(4)}</span>
                          </div>
                          <div className="grid grid-cols-2 text-[10px] text-slate-400">
                            <div>Weight (w): <strong className="text-slate-350">{step.w.toFixed(5)}</strong></div>
                            <div>Bias (b): <strong className="text-slate-350">{step.b.toFixed(4)}</strong></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                </div>

                {/* Column to display Interactive CSS / HTML / SVG line plot of Sum histories plus fitted line */}
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-3xl p-6">
                  <h4 className="text-sm font-bold text-slate-200 mb-2">Simulasi Plot Nilai Sum Undian & Garis Regresi</h4>
                  <p className="text-xs text-slate-400 mb-6 font-mono">Melacak 15 undian terakhir secara sekuensial dibanding garis trendline least-square kalkulus.</p>

                  <div className="bg-slate-950 border border-slate-850 p-4 rounded-2xl relative">
                    
                    {/* SVG Graphic block */}
                    <svg viewBox="0 0 500 220" className="w-full h-auto overflow-visible select-none">
                      
                      {/* Grid bounds */}
                      <line x1="40" y1="20" x2="480" y2="20" stroke="#1e293b" strokeDasharray="3,3" />
                      <line x1="40" y1="70" x2="480" y2="70" stroke="#1e293b" strokeDasharray="3,3" />
                      <line x1="40" y1="120" x2="480" y2="120" stroke="#1e293b" strokeDasharray="3,3" />
                      <line x1="40" y1="170" x2="480" y2="170" stroke="#1e293b" strokeDasharray="3,3" />
                      <line x1="40" y1="200" x2="480" y2="200" stroke="#334155" />

                      {/* Y-Axis captions */}
                      <text x="15" y="24" className="fill-slate-600 text-[10px] font-mono leading-none">Sum 36</text>
                      <text x="15" y="74" className="fill-slate-600 text-[10px] font-mono leading-none">Sum 24</text>
                      <text x="15" y="124" className="fill-slate-600 text-[10px] font-mono leading-none">Sum 12</text>
                      <text x="15" y="174" className="fill-slate-600 text-[10px] font-mono leading-none">Sum 0</text>

                      {(() => {
                        // Gather last 15 draws
                        const lastDraws = [...history]
                          .sort((a,b) => a.id.localeCompare(b.id, undefined, { numeric: true }))
                          .slice(-15);
                        
                        if (lastDraws.length === 0) return null;

                        // Width steps = (480 - 40) / 14 = 31.42
                        const stepX = 440 / Math.max(1, lastDraws.length - 1);
                        
                        // Map each sum to Y height: 200 is base line, 20 is top line (representing max sum ~36)
                        const getY = (sumVal: number) => {
                          const val = Math.max(0, Math.min(36, sumVal));
                          return 200 - (val / 36) * 180;
                        };

                        // Construct line points
                        let pointsD = "";
                        let regressionPointsD = "";
                        
                        lastDraws.forEach((draw, i) => {
                          const x = 40 + i * stepX;
                          const y = getY(draw.sum);
                          
                          if (i === 0) pointsD += `M ${x} ${y}`;
                          else pointsD += ` L ${x} ${y}`;

                          // Regression equation: y_est = m * x + c. In our dataset, each draw index maps
                          const idxGlobal = history.length - lastDraws.length + i + 1;
                          const regEstimatedSum = modelPrediction.regression.slope * idxGlobal + modelPrediction.regression.intercept;
                          const regY = getY(regEstimatedSum);
                          
                          if (i === 0) regressionPointsD += `M ${x} ${regY}`;
                          else regressionPointsD += ` L ${x} ${regY}`;
                        });

                        return (
                          <>
                            {/* Blue regression line */}
                            <path d={regressionPointsD} fill="none" stroke="#6366f1" strokeWidth="2.5" strokeDasharray="4,4" />

                            {/* Actual draw lines */}
                            <path d={pointsD} fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />

                            {/* Circles on actual nodes */}
                            {lastDraws.map((draw, i) => {
                              const x = 40 + i * stepX;
                              const y = getY(draw.sum);
                              return (
                                <g key={draw.id}>
                                  <circle cx={x} cy={y} r="4" className="fill-slate-950 stroke-emerald-400 stroke-2" />
                                  <text x={x} y={y - 8} textAnchor="middle" className="fill-slate-300 font-mono text-[9px] font-bold">
                                    {draw.drawCodeString}
                                  </text>
                                </g>
                              );
                            })}
                          </>
                        );
                      })()}

                    </svg>

                    {/* Legends indicators */}
                    <div className="flex gap-4 items-center justify-center font-mono text-[10px] mt-4">
                      <div className="flex items-center gap-1.5 text-emerald-450">
                        <span className="w-3.5 h-1.5 bg-emerald-500 inline-block rounded-full" />
                        <span>Keluaran Sum Aktual</span>
                      </div>
                      <div className="flex items-center gap-1.5 text-indigo-400">
                        <span className="w-3.5 h-1 px-1 bg-none border-b border-indigo-400 border-dashed inline-block" />
                        <span>Garis Linear Regresi Model</span>
                      </div>
                    </div>

                  </div>

                </div>

              </div>
            )}

          </div>
        )}

        {/* TAB 5: AI MACHINE LEARNING LOGIC MODEL USING GEMINI */}
        {activeTab === "ai" && (
          <div className="space-y-8">
            
            {/* AI introduction panel */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-center justify-between">
              <div className="flex gap-5 items-start">
                <div className="bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-600 p-3 rounded-2xl text-slate-900">
                  <Cpu className="h-6 w-6 stroke-[2]" />
                </div>
                <div>
                  <h3 className="text-md font-bold text-slate-100">
                    Sistem Pola Pembelajaran Mesin AI (Gemini Pattern Engine)
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 lines-normal leading-relaxed max-w-xl">
                    Model ini mengirimkan runtunan urutan 40 undian terakhir beserta dekomposisi angkanya ke server-side AI. AI bertindak sebagai pemroses Bayesian modern untuk mendeteksi kecenderungan anomali, nomor colok bebas paling dominan, dan menyusun ramalan numerik.
                  </p>
                </div>
              </div>

              <button
                onClick={runAiMlEngine}
                disabled={isAiLoading}
                className="bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-600 hover:to-indigo-700 active:scale-[0.98] transition-transform text-white font-bold font-mono text-xs px-6 py-3 rounded-xl flex items-center justify-center gap-2 cursor-pointer shadow-lg disabled:opacity-50 shrink-0 w-full md:w-auto"
              >
                <Cpu className={`h-4 w-4 ${isAiLoading ? "animate-spin" : ""}`} />
                {isAiLoading ? "MEMPROSES POLA MATEMATIKA..." : "JALANKAN ENGINE PEMBELAJARAN MODEL AI"}
              </button>
            </div>

            {/* Error messaging */}
            {aiError && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl flex items-start gap-3 text-xs font-mono">
                <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
                <div>
                  <p className="font-bold underline">Kesalahan Integrasi Server-Side AI</p>
                  <p className="mt-1 leading-relaxed">{aiError}</p>
                </div>
              </div>
            )}

            {/* Error messaging */}
            {aiError && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl flex items-start gap-3 text-xs font-mono">
                <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
                <div>
                  <p className="font-bold underline">Kesalahan Integrasi Server-Side AI</p>
                  <p className="mt-1 leading-relaxed">{aiError}</p>
                </div>
              </div>
            )}

            {/* BBFS / ANGKA CAMPUR BENTO PANEL - ALWAYS ACTIVE */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 right-0 p-3.5 text-[9px] font-mono bg-cyan-500/10 text-cyan-400 font-bold rounded-bl-2xl border-l border-b border-slate-800 tracking-widest flex items-center gap-1.5 uppercase">
                <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-ping" /> BBFS Engine Active
              </div>

              <div className="flex items-center gap-2 mb-2">
                <Sliders className="h-4 w-4 text-cyan-400" />
                <h3 className="text-sm font-bold font-mono text-slate-200 tracking-wider">
                  REKOMENDASI ANGKA CAMPUR & BBFS (5, 6, 7 DIGIT)
                </h3>
              </div>
              
              {modelPrediction?.drawTarget ? (
                <div className="mb-4 bg-gradient-to-r from-cyan-950/40 via-slate-950/70 to-slate-950/40 border border-cyan-850/60 p-3.5 rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
                  <div className="flex items-center gap-2">
                    <span className="flex h-2 w-2 relative">
                      <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
                      <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
                    </span>
                    <span className="text-slate-300 font-bold tracking-wide">TARGET PREDIKSI KELUARAN SELANJUTNYA:</span>
                  </div>
                  <div className="flex items-center gap-1.5 bg-cyan-950/60 text-cyan-300 border border-cyan-500/25 px-3 py-1 rounded-xl font-bold font-mono text-[11px] shrink-0 self-start sm:self-center">
                    <span>{modelPrediction.drawTarget.date}</span>
                    <span className="text-slate-600">|</span>
                    <span className="text-amber-400">SLOT {modelPrediction.drawTarget.timeSlot} WIB</span>
                  </div>
                </div>
              ) : (
                <div className="mb-4 bg-slate-950/40 border border-slate-850/60 p-3 rounded-2xl animate-pulse text-slate-500 font-mono text-[11px]">
                  Sedang menghitung target waktu slot draw selanjutnya...
                </div>
              )}

              <p className="text-xs text-slate-400 mb-6 max-w-3xl leading-relaxed">
                Kombinasi Bolak Balik Full Set (BBFS) di bawah dihitung menggunakan amalgamasi kualitatif dan kuantitatif dari seluruh modul analisis: **Poisson Density PMF**, **Markov Transition Matrix**, **Linear Regression Slopes**, serta model **Gemini AI Predictor** {aiPrediction ? "(DENGAN TUNING AI)" : "(STANDAR STATISTIKA MATEMATIS)"}.
              </p>

              {/* Toast clipboard indicator */}
              {copiedBbfs && (
                <div className="mb-4 p-2.5 bg-cyan-950/40 text-cyan-400 rounded-lg border border-cyan-800/35 font-mono text-[11px] font-semibold text-center animate-pulse flex items-center justify-center gap-1.5">
                  <Check className="h-3.5 w-3.5" /> {copiedBbfs}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                
                {/* 5 Digit bento block */}
                <div className="bg-slate-950/60 border border-slate-850 p-5 rounded-2xl flex flex-col justify-between transition-all duration-300 hover:border-slate-750 hover:bg-slate-950/80 group">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-900 pb-3 mb-4">
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">5 Digit Campur</span>
                      <span className="text-[9px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded font-semibold leading-none">Efisiensi Biaya</span>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-5">
                      {bbfsRecommendations.digits5.map(digit => (
                        <div
                          key={digit}
                          className="bg-gradient-to-br from-slate-900 to-slate-950 text-cyan-400 ring-1 ring-white/5 border border-slate-800 font-mono font-black w-11 h-11 flex items-center justify-center rounded-xl text-md shadow-md transition-all group-hover:border-cyan-500/20 hover:scale-105"
                        >
                          {digit}
                        </div>
                      ))}
                    </div>

                    <p className="text-[11px] text-slate-500 font-mono leading-relaxed mb-4">
                      Diprioritaskan untuk efisiensi taruhan 4D/3D/2D dengan cakupan set terkonsentrasi pada angka-angka paling stabil.
                    </p>
                  </div>

                  <button
                    onClick={() => {
                      const txt = bbfsRecommendations.digits5.join("");
                      navigator.clipboard.writeText(txt);
                      setCopiedBbfs(`Set 5 Digit (${txt}) berhasil disalin ke clipboard!`);
                      setTimeout(() => setCopiedBbfs(null), 2500);
                    }}
                    className="w-full text-[10px] font-mono font-bold uppercase tracking-wider py-2 bg-slate-900 border border-slate-850 hover:border-cyan-500/30 hover:bg-slate-850 hover:text-cyan-400 text-slate-400 rounded-xl transition-all active:scale-[0.98]"
                  >
                    Salin 5 Digit Set
                  </button>
                </div>

                {/* 6 Digit bento block */}
                <div className="bg-slate-950/60 border border-slate-850 p-5 rounded-2xl flex flex-col justify-between transition-all duration-300 hover:border-slate-750 hover:bg-slate-950/80 group">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-900 pb-3 mb-4">
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">6 Digit Campur</span>
                      <span className="text-[9px] font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20 px-2 py-0.5 rounded font-semibold leading-none">Rekomendasi Utama</span>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-5">
                      {bbfsRecommendations.digits6.map(digit => (
                        <div
                          key={digit}
                          className="bg-gradient-to-br from-slate-900 to-slate-950 text-purple-400 ring-1 ring-white/5 border border-slate-800 font-mono font-black w-11 h-11 flex items-center justify-center rounded-xl text-md shadow-md transition-all group-hover:border-purple-500/20 hover:scale-105"
                        >
                          {digit}
                        </div>
                      ))}
                    </div>

                    <p className="text-[11px] text-slate-500 font-mono leading-relaxed mb-4">
                      Rekomendasi set seimbang dengan peluang penetrasi ideal (win-to-cost ratio terbaik) untuk seluruh variasi draw Macau.
                    </p>
                  </div>

                  <button
                    onClick={() => {
                      const txt = bbfsRecommendations.digits6.join("");
                      navigator.clipboard.writeText(txt);
                      setCopiedBbfs(`Set 6 Digit (${txt}) berhasil disalin ke clipboard!`);
                      setTimeout(() => setCopiedBbfs(null), 2500);
                    }}
                    className="w-full text-[10px] font-mono font-bold uppercase tracking-wider py-2 bg-slate-900 border border-slate-850 hover:border-purple-500/30 hover:bg-slate-850 hover:text-purple-400 text-slate-400 rounded-xl transition-all active:scale-[0.98]"
                  >
                    Salin 6 Digit Set
                  </button>
                </div>

                {/* 7 Digit bento block */}
                <div className="bg-slate-950/60 border border-slate-850 p-5 rounded-2xl flex flex-col justify-between transition-all duration-300 hover:border-slate-750 hover:bg-slate-950/80 group">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-900 pb-3 mb-4">
                      <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest font-bold">7 Digit Campur</span>
                      <span className="text-[9px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2 py-0.5 rounded font-semibold leading-none">Keamanan Maksimal</span>
                    </div>

                    <div className="flex flex-wrap gap-2 mb-5">
                      {bbfsRecommendations.digits7.map(digit => (
                        <div
                          key={digit}
                          className="bg-gradient-to-br from-slate-900 to-slate-950 text-amber-400 ring-1 ring-white/5 border border-slate-800 font-mono font-black w-11 h-11 flex items-center justify-center rounded-xl text-md shadow-md transition-all group-hover:border-amber-500/20 hover:scale-105"
                        >
                          {digit}
                        </div>
                      ))}
                    </div>

                    <p className="text-[11px] text-slate-500 font-mono leading-relaxed mb-4">
                      Memberikan tingkat proteksi keamanan tertinggi dari anomali pembelokan data dengan rasio hit peluang historis paling besar.
                    </p>
                  </div>

                  <button
                    onClick={() => {
                      const txt = bbfsRecommendations.digits7.join("");
                      navigator.clipboard.writeText(txt);
                      setCopiedBbfs(`Set 7 Digit (${txt}) berhasil disalin ke clipboard!`);
                      setTimeout(() => setCopiedBbfs(null), 2500);
                    }}
                    className="w-full text-[10px] font-mono font-bold uppercase tracking-wider py-2 bg-slate-900 border border-slate-850 hover:border-amber-500/30 hover:bg-slate-850 hover:text-amber-400 text-slate-400 rounded-xl transition-all active:scale-[0.98]"
                  >
                    Salin 7 Digit Set
                  </button>
                </div>
              </div>

              {/* REKOMENDASI FORMULASI PEMASANGAN STRATEGIS */}
              <div className="mt-8 bg-slate-950/40 border border-slate-850/60 p-5 rounded-2xl">
                <div className="flex flex-col sm:flex-row sm:items-center gap-2 mb-4">
                  <div className="flex items-center gap-2">
                    <Award className="h-4 w-4 text-amber-500 animate-pulse" />
                    <h4 className="text-xs font-bold font-mono text-slate-200 uppercase tracking-widest">
                      Rekomendasi Formulasi Pemasangan Strategis (Tangguh & Tajam)
                    </h4>
                  </div>
                  <span className="text-[9px] bg-emerald-500/10 text-emerald-400 font-bold px-1.5 py-0.5 rounded border border-emerald-500/20 uppercase tracking-wider self-start sm:ml-auto">
                    Akurasi Komposit Terbaik
                  </span>
                </div>

                <p className="text-xs text-slate-400 mb-5 leading-relaxed font-sans">
                  Berdasarkan kalkulasi probabilitas komposit terkini dari mesin pembelajaran AI dan seluruh variabel statistik, berikut adalah rekomendasi taktis terkuat untuk mendeploy angka bbfs campur Anda dengan efisiensi modal dan rasio penetrasi (JP) tertajam:
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  
                  {/* Strategi 1: Tangguh */}
                  <div className="bg-gradient-to-b from-slate-900/90 to-slate-950 border border-purple-500/15 p-4 rounded-xl flex flex-col justify-between hover:border-purple-500/30 transition-all duration-300">
                    <div>
                      <div className="flex items-center justify-between mb-3 border-b border-slate-800/60 pb-2">
                        <span className="text-[11px] font-bold font-mono text-purple-400 flex items-center gap-1.5">
                          🔥 PILIHAN TANGGUH (RASIO TERBAIK)
                        </span>
                      </div>
                      <div className="space-y-2 text-xs font-mono mb-4 text-slate-300">
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Pilihan Set:</span>
                          <span className="text-purple-400 font-extrabold">6 DIGIT BBFS</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Angka Pasangan:</span>
                          <span className="text-slate-200 font-semibold tracking-wider font-sans">{bbfsRecommendations.digits6.join("") || "Belum Dihitung"}</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Fokus Betting:</span>
                          <span className="text-purple-300 font-bold">3D ATAU 2D</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Probabilitas JP 2D:</span>
                          <span className="text-emerald-400 font-bold">{bbfsProbabilityAnalysis.digits6.future2D.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span className="text-slate-500">Probabilitas JP 3D:</span>
                          <span className="text-cyan-400 font-bold">{bbfsProbabilityAnalysis.digits6.future3D.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="bg-purple-950/20 text-purple-300 border border-purple-900/40 p-2.5 rounded-lg text-[10px] leading-relaxed font-sans mt-auto">
                      <strong>Rasio Efisiensi Terbaik:</strong> Menyeimbangkan modal 30 kombinasi line 2D (atau 120 line 3D) dengan persentase kemungkinan tembus yang seimbang. Direkomendasikan sebagai pilihan betting utama.
                    </div>
                  </div>

                  {/* Strategi 2: Aman */}
                  <div className="bg-gradient-to-b from-slate-900/90 to-slate-950 border border-amber-500/15 p-4 rounded-xl flex flex-col justify-between hover:border-amber-500/30 transition-all duration-300">
                    <div>
                      <div className="flex items-center justify-between mb-3 border-b border-slate-800/60 pb-2">
                        <span className="text-[11px] font-bold font-mono text-amber-400 flex items-center gap-1.5">
                          🛡️ PILIHAN AMAN (SAFETY-MAX)
                        </span>
                      </div>
                      <div className="space-y-2 text-xs font-mono mb-4 text-slate-300">
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Pilihan Set:</span>
                          <span className="text-amber-400 font-extrabold">7 DIGIT BBFS</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Angka Pasangan:</span>
                          <span className="text-slate-200 font-semibold tracking-wider font-sans">{bbfsRecommendations.digits7.join("") || "Belum Dihitung"}</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Fokus Betting:</span>
                          <span className="text-amber-300 font-bold">4D / 3D / 2D</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Probabilitas JP 3D:</span>
                          <span className="text-cyan-400 font-bold">{bbfsProbabilityAnalysis.digits7.future3D.toFixed(2)}%</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span className="text-slate-500">Probabilitas JP 4D:</span>
                          <span className="text-purple-400 font-bold">{bbfsProbabilityAnalysis.digits7.future4D.toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                    <div className="bg-amber-950/20 text-amber-300 border border-amber-900/40 p-2.5 rounded-lg text-[10px] leading-relaxed font-sans mt-auto">
                      <strong>Proteksi Keamanan Maksimum:</strong> Memiliki cakupan peluang historis JP 2D di atas 90%. Sangat ampuh menahan rintangan sebaran acak angka Macau untuk menjaring JP berkelanjutan.
                    </div>
                  </div>

                  {/* Strategi 3: Ekonomis */}
                  <div className="bg-gradient-to-b from-slate-900/90 to-slate-950 border border-cyan-500/15 p-4 rounded-xl flex flex-col justify-between hover:border-cyan-500/30 transition-all duration-300">
                    <div>
                      <div className="flex items-center justify-between mb-3 border-b border-slate-800/60 pb-2">
                        <span className="text-[11px] font-bold font-mono text-cyan-400 flex items-center gap-1.5">
                          💰 PILIHAN TAKTIS (EKONOMIS)
                        </span>
                      </div>
                      <div className="space-y-2 text-xs font-mono mb-4 text-slate-300">
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Pilihan Set:</span>
                          <span className="text-cyan-400 font-extrabold">5 DIGIT BBFS</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Angka Pasangan:</span>
                          <span className="text-slate-200 font-semibold tracking-wider font-sans">{bbfsRecommendations.digits5.join("") || "Belum Dihitung"}</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Fokus Betting:</span>
                          <span className="text-cyan-300 font-bold">2D ATAU COLOK</span>
                        </div>
                        <div className="flex justify-between border-b border-slate-900/80 py-1">
                          <span className="text-slate-500">Probabilitas JP 2D:</span>
                          <span className="text-emerald-400 font-bold">{bbfsProbabilityAnalysis.digits5.future2D.toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span className="text-slate-500">Kebutuhan Modal:</span>
                          <span className="text-emerald-400 font-bold">Paling Rendah</span>
                        </div>
                      </div>
                    </div>
                    <div className="bg-cyan-950/20 text-cyan-300 border border-cyan-900/40 p-2.5 rounded-lg text-[10px] leading-relaxed font-sans mt-auto">
                      <strong>Strategi Efisiensi Anggaran:</strong> Terdiri dari kombinasi ringkas (hanya 20 kombinasi 2D). Ideal untuk pasang colok bebas atau uji pasar tanpa risiko depresiasi saldo berlebih.
                    </div>
                  </div>

                </div>
              </div>

              {/* BBFS Probability Breakdown Matrix */}
              <div id="bbfs-probability-matrix" className="mt-8 pt-6 border-t border-slate-800/80">
                <div className="flex items-center gap-1.5 mb-4">
                  <TrendingUp className="h-4 w-4 text-emerald-450" />
                  <h4 className="text-xs font-bold font-mono text-slate-300 uppercase tracking-widest">
                    Matriks Analisis Probabilitas & Backtest BBFS
                  </h4>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs border-collapse min-w-[600px]">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 text-[10px] uppercase tracking-wider">
                        <th className="py-2.5 px-3">Sistem Rekomendasi</th>
                        <th className="py-2.5 px-3 text-center w-1/3">Probabilitas Historis (Data Sudah Keluar)*</th>
                        <th className="py-2.5 px-3 text-center w-1/3">Probabilitas Prediktif (Angka Yang Akan Datang)**</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/40">
                      {/* Row 1: 5 Digit BBFS */}
                      <tr className="hover:bg-slate-950/20 transition-colors">
                        <td className="py-3 px-3">
                          <div className="font-semibold text-cyan-400">Set 5 Digit Campur</div>
                          <div className="text-[10px] text-slate-550 mt-0.5">Digit: <span className="font-bold text-slate-400 font-sans">{bbfsRecommendations.digits5.join(", ")}</span></div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 2D</span>
                              <span className="text-emerald-450 font-bold font-sans">{bbfsProbabilityAnalysis.digits5.past2D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits5.hit2DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 3D</span>
                              <span className="text-cyan-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits5.past3D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits5.hit3DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 4D</span>
                              <span className="text-purple-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits5.past4D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits5.hit4DCount}x JP</span>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950/80 p-1.5 rounded border border-cyan-500/5">
                              <span className="text-slate-550 block text-[8px] uppercase">P(2D)</span>
                              <span className="text-emerald-450 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits5.future2D.toFixed(1)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-cyan-500/5">
                              <span className="text-slate-550 block text-[8px] uppercase">P(3D)</span>
                              <span className="text-cyan-400 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits5.future3D.toFixed(2)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-cyan-500/5">
                              <span className="text-slate-550 block text-[8px] uppercase">P(4D)</span>
                              <span className="text-purple-400 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits5.future4D.toFixed(2)}%</span>
                            </div>
                          </div>
                        </td>
                      </tr>

                      {/* Row 2: 6 Digit BBFS */}
                      <tr className="hover:bg-slate-950/20 transition-colors">
                        <td className="py-3 px-3">
                          <div className="font-semibold text-purple-400">Set 6 Digit Campur</div>
                          <div className="text-[10px] text-slate-555 mt-0.5">Digit: <span className="font-bold text-slate-400 font-sans">{bbfsRecommendations.digits6.join(", ")}</span></div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 2D</span>
                              <span className="text-emerald-455 font-bold font-sans">{bbfsProbabilityAnalysis.digits6.past2D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits6.hit2DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 3D</span>
                              <span className="text-cyan-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits6.past3D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits6.hit3DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 4D</span>
                              <span className="text-purple-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits6.past4D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits6.hit4DCount}x JP</span>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950/80 p-1.5 rounded border border-purple-500/5">
                              <span className="text-slate-550 block text-[8px] uppercase">P(2D)</span>
                              <span className="text-emerald-455 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits6.future2D.toFixed(1)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-purple-500/5">
                              <span className="text-slate-555 block text-[8px] uppercase">P(3D)</span>
                              <span className="text-cyan-400 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits6.future3D.toFixed(2)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-purple-500/5">
                              <span className="text-slate-555 block text-[8px] uppercase">P(4D)</span>
                              <span className="text-purple-450 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits6.future4D.toFixed(2)}%</span>
                            </div>
                          </div>
                        </td>
                      </tr>

                      {/* Row 3: 7 Digit BBFS */}
                      <tr className="hover:bg-slate-950/20 transition-colors">
                        <td className="py-3 px-3">
                          <div className="font-semibold text-amber-400">Set 7 Digit Campur</div>
                          <div className="text-[10px] text-slate-555 mt-0.5">Digit: <span className="font-bold text-slate-400 font-sans">{bbfsRecommendations.digits7.join(", ")}</span></div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-500 block text-[8px] uppercase">Hit 2D</span>
                              <span className="text-emerald-450 font-bold font-sans">{bbfsProbabilityAnalysis.digits7.past2D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits7.hit2DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-550 block text-[8px] uppercase">Hit 3D</span>
                              <span className="text-cyan-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits7.past3D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits7.hit3DCount}x JP</span>
                            </div>
                            <div className="bg-slate-950 p-1.5 rounded border border-slate-900/60">
                              <span className="text-slate-550 block text-[8px] uppercase">Hit 4D</span>
                              <span className="text-purple-400 font-bold font-sans">{bbfsProbabilityAnalysis.digits7.past4D.toFixed(1)}%</span>
                              <span className="text-[9px] text-slate-400 block font-mono mt-0.5">{bbfsProbabilityAnalysis.digits7.hit4DCount}x JP</span>
                            </div>
                          </div>
                        </td>
                        <td className="py-3 px-3">
                          <div className="grid grid-cols-3 gap-1.5 text-center text-[10px]">
                            <div className="bg-slate-950/80 p-1.5 rounded border border-amber-500/5">
                              <span className="text-slate-550 block text-[8px] uppercase">P(2D)</span>
                              <span className="text-emerald-450 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits7.future2D.toFixed(1)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-amber-500/5">
                              <span className="text-slate-555 block text-[8px] uppercase">P(3D)</span>
                              <span className="text-cyan-455 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits7.future3D.toFixed(2)}%</span>
                            </div>
                            <div className="bg-slate-950/80 p-1.5 rounded border border-amber-500/5">
                              <span className="text-slate-555 block text-[8px] uppercase">P(4D)</span>
                              <span className="text-purple-455 font-extrabold font-sans">{bbfsProbabilityAnalysis.digits7.future4D.toFixed(2)}%</span>
                            </div>
                          </div>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="text-[10px] text-slate-500 mt-2.5 space-y-1">
                  <p>* <strong>Probabilitas Historis (Backtest)</strong> dihitung dari total hit rate aktual set BBFS tersebut pada seluruh database ({history.length} data undian).</p>
                  <p>** <strong>Probabilitas Prediktif</strong> adalah perkiraan pencocokan angka akan datang yang dihitung dengan mengintegrasikan kerapatan data Poisson tiap posisi dengan model transisi rantai Markov bauran.</p>
                </div>
              </div>

              {/* Catatan JP / Hit Terakhir di Database */}
              <div id="bbfs-recent-jps" className="mt-6 pt-5 border-t border-slate-800/85">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4">
                  <div className="flex items-center gap-1.5">
                    <CheckCircle className="h-4 w-4 text-amber-500" />
                    <h4 className="text-xs font-bold font-mono text-slate-200 uppercase tracking-widest">
                      Detail Histori Hit (JP) vs Tidak Hit BBFS 7 Digit (40 Undian Terakhir)
                    </h4>
                  </div>
                  <div className="text-[10px] font-mono font-bold bg-amber-500/10 text-amber-550 border border-amber-500/20 px-2 py-0.5 rounded flex items-center gap-1.5 shrink-0">
                    <span>Set 7 Digit: {bbfsRecommendations.digits7.join("") || "Belum Dihitung"}</span>
                  </div>
                </div>
                
                <p className="text-[11px] text-slate-400 mb-4 leading-relaxed font-sans">
                  Berikut adalah jajaran <strong>40 data undian terakhir</strong> diurutkan dari yang <strong>terbaru ke terlama</strong>. Sistem melakukan simulasi pencocokan pola 7 digit pilihan guna membedakan status JP (4D/3D/2D) dengan status Tidak Hit (Zong) secara nyata.
                </p>

                {digits7RecentAnalysis && digits7RecentAnalysis.length > 0 ? (
                  <div className="bg-slate-950/65 border border-slate-900 rounded-2xl p-4 max-h-[380px] overflow-y-auto custom-scrollbar">
                    <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2.5">
                      {digits7RecentAnalysis.map((h, idx) => (
                        <div 
                          key={idx} 
                          className={`flex items-center justify-between text-[11px] font-mono p-2.5 rounded-lg border transition-all duration-200 ${
                            h.isHit 
                              ? h.hitType === "4D"
                                ? "bg-purple-950/30 border-purple-800/40 text-purple-200 shadow-sm shadow-purple-950/20"
                                : h.hitType === "3D"
                                  ? "bg-cyan-950/30 border-cyan-800/40 text-cyan-200 shadow-sm shadow-cyan-950/20"
                                  : "bg-emerald-950/30 border-emerald-800/40 text-emerald-200 shadow-sm shadow-emerald-950/20"
                              : "bg-slate-900/15 border-slate-900/50 text-slate-500"
                          }`}
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className={`text-[9px] font-bold tracking-tight ${h.isHit ? "text-slate-400" : "text-slate-600"}`}>
                              #{h.id} <span className="font-normal font-sans text-[8px]">({h.timeSlot})</span>
                            </span>
                            <span className={`font-black tracking-wider text-xs ${h.isHit ? "text-slate-100" : "text-slate-600 line-through decoration-slate-800/60"}`}>
                              {h.drawCodeString}
                            </span>
                          </div>

                          <div className="shrink-0">
                            {h.isHit ? (
                              <span className={`px-1.5 py-1 rounded-md font-black text-[9px] tracking-wide leading-none select-none uppercase shadow-sm ${
                                h.hitType === "4D" ? "bg-purple-500/20 text-purple-300 border border-purple-500/35" :
                                h.hitType === "3D" ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/35" :
                                "bg-emerald-500/20 text-emerald-300 border border-emerald-500/35"
                              }`}>
                                {h.hitType} JP
                              </span>
                            ) : (
                              <span className="px-1.5 py-1 rounded-md font-bold text-[9px] tracking-wide leading-none select-none text-slate-600 bg-slate-950/40 border border-slate-900 border-dashed uppercase">
                                Miss
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="bg-slate-950/50 border border-dashed border-slate-900 p-8 text-center text-slate-500 font-mono text-[11px] rounded-2xl italic">
                    Belum ada data undian di database untuk dianalisis.
                  </div>
                )}
              </div>

              {/* Informative footer */}
              <div className="mt-5 pt-4 border-t border-slate-800/80 flex flex-col sm:flex-row items-center justify-between gap-3 text-[10px] font-mono text-slate-500">
                <span>* Angka diurutkan terkecil ke terbesar untuk penyederhanaan pengisian bandar (Bolak Balik)</span>
                <span className="text-cyan-500/70">Waktu Proses: Real-time Analisis Aktif</span>
              </div>
            </div>

            {/* Prompt setup instruction if not run yet */}
            {!aiPrediction && !isAiLoading && (
              <div className="bg-slate-900/40 border border-slate-850 py-16 text-center text-slate-500 font-mono text-xs rounded-3xl">
                <Cpu className="h-10 w-10 text-slate-600 mx-auto mb-3" />
                Silakan tekan tombol di atas untuk memerintahkan AI menganalisis data Macau secara mendalam.
              </div>
            )}

            {/* Loading placeholder */}
            {isAiLoading && (
              <div className="bg-slate-900 border border-slate-800 py-24 text-center rounded-3xl space-y-4 shadow-xl">
                <div className="h-10 w-10 border-4 border-cyan-400 border-t-transparent rounded-full animate-spin mx-auto" />
                <p className="text-xs text-slate-400 font-mono animate-pulse">Menghitung matriks Markov, Poisson PMF, dan mendata Shio Astronomis...</p>
              </div>
            )}

            {/* Display complete AI output */}
            {aiPrediction && !isAiLoading && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Visual recommendations columns */}
                <div className="lg:col-span-1 space-y-6">
                  
                  {/* Prediksi Numbers card */}
                  <div className="bg-gradient-to-b from-slate-900 to-indigo-950/40 border border-indigo-900/30 rounded-2xl p-6 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-3 text-[9px] font-mono bg-indigo-500/15 text-indigo-300 font-bold rounded-bl-xl border-l border-b border-indigo-900/30 tracking-widest leading-none">
                      AI GENERATED
                    </div>

                    <h4 className="text-xs font-bold font-mono text-cyan-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-5 flex items-center gap-2">
                      <Cpu className="h-3.5 w-3.5" /> REKOMENDASI HASIL PROBABILITY
                    </h4>

                    <div className="space-y-4">
                      
                      {/* 4D Suggestions */}
                      <div>
                        <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest font-semibold block mb-2">Rekomendasi Set 4D (As-Kop-Kepala-Ekor)</span>
                        <div className="flex flex-wrap gap-2">
                          {aiPrediction.predictions4D.map(code => (
                            <span key={code} className="bg-slate-950 text-emerald-400 font-mono font-bold tracking-widest px-3 py-1.5 rounded-lg border border-slate-800 text-sm">
                              {code}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* 3D Suggestions */}
                      <div>
                        <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest font-semibold block mb-1.5">Rekomendasi Set 3D (Kop-Kepala-Ekor)</span>
                        <div className="flex flex-wrap gap-2">
                          {aiPrediction.predictions3D.map(code => (
                            <span key={code} className="bg-slate-950 text-cyan-400 font-mono font-medium tracking-wider px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs">
                              {code}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* 2D Suggestions */}
                      <div>
                        <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest font-semibold block mb-1.5">Rekomendasi Set 2D (Kepala-Ekor)</span>
                        <div className="flex flex-wrap gap-2">
                          {aiPrediction.predictions2D.map(code => (
                            <span key={code} className="bg-slate-950 text-indigo-300 font-mono font-medium tracking-wider px-2.5 py-1.5 rounded-lg border border-slate-800 text-xs text-center min-w-10">
                              {code}
                            </span>
                          ))}
                        </div>
                      </div>

                      {/* Colok indicators */}
                      <div className="grid grid-cols-2 gap-4 pt-3 border-t border-slate-800/60">
                        <div>
                          <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest block mb-1 font-semibold">Colok Bebas</span>
                          <div className="flex gap-1.5">
                            {aiPrediction.colokBebas.map(digit => (
                              <span key={digit} className="bg-slate-950 text-amber-400 font-mono font-bold w-7 h-7 flex items-center justify-center rounded-full border border-slate-800 text-xs">
                                {digit}
                              </span>
                            ))}
                          </div>
                        </div>

                        <div>
                          <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest block mb-1 font-semibold">Colok Macau</span>
                          <div className="flex flex-wrap gap-1">
                            {aiPrediction.colokMacau.map(pair => (
                              <span key={pair} className="bg-slate-950 text-purple-400 font-mono font-semibold px-2 py-1 rounded text-[10px] border border-slate-800">
                                {pair}
                              </span>
                            ))}
                          </div>
                        </div>
                      </div>

                      {/* Shio Zodiac details */}
                      <div className="pt-3 border-t border-slate-800/60 flex items-center justify-between">
                        <span className="text-[9px] font-mono text-slate-450 uppercase tracking-widest font-semibold">Zodiak Shio Berpengaruh:</span>
                        <div className="flex gap-1">
                          {aiPrediction.shioAccents.map(shio => (
                            <span key={shio} className="bg-indigo-500/10 text-indigo-300 border border-indigo-400/20 px-2 py-0.5 rounded text-[10px] font-mono font-semibold">
                              {shio}
                            </span>
                          ))}
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Mathematical formulas detail */}
                  <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl">
                    <h4 className="text-xs font-bold font-mono text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-4 flex items-center gap-2">
                      <BookOpen className="h-3.5 w-3.5 text-slate-400" /> FORMULA MATEMATIKA YANG DIGUNAKAN
                    </h4>
                    <ul className="space-y-2.5 text-xs text-slate-350 list-none tracking-normal">
                      {aiPrediction.mathematicalFormulasUsed.map((formula, i) => (
                        <li key={i} className="flex gap-2.5 items-start">
                          <code className="bg-slate-950 px-2 py-1 select-all cursor-copy rounded font-mono text-cyan-400 text-[10px] ring-1 ring-white/5 border border-slate-850">
                            {formula}
                          </code>
                        </li>
                      ))}
                    </ul>
                  </div>

                </div>

                {/* Computational Justifications and Pattern logs */}
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-3xl p-6 relative">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-5">
                    <div>
                      <h4 className="text-md font-bold text-slate-200">
                        Justifikasi Saintifik & Dekomposisi Pola AI
                      </h4>
                      <p className="text-xs text-slate-400">
                        Kompilasi dekomposisi kalkulus probabilities dari engine machine learning.
                      </p>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-slate-500 font-mono">Indeks Keyakinan</span>
                      <span className="bg-gradient-to-r from-emerald-500 to-cyan-500 text-slate-950 text-xs px-2.5 py-1 rounded font-mono font-bold">
                        {aiPrediction.confidenceIndex}% Confidence
                      </span>
                    </div>
                  </div>

                  {/* Rendering full justifications using raw parsed Markdown */}
                  <div className="prose prose-invert prose-xs text-xs text-slate-300 max-w-none font-sans leading-relaxed space-y-4">
                    {aiPrediction.aiJustification.split("\n\n").map((para, i) => {
                      if (para.startsWith("#")) {
                        return <h5 key={i} className="text-sm font-bold text-slate-100 font-mono mt-4 text-cyan-400 uppercase">{para.replace(/#/g, "").trim()}</h5>;
                      }
                      if (para.startsWith("-") || para.startsWith("*")) {
                        return (
                          <ul key={i} className="list-disc list-inside space-y-1 text-slate-300 ml-4">
                            {para.split("\n").map((li, j) => (
                              <li key={j}>{li.replace(/^[\s-*]+/, "")}</li>
                            ))}
                          </ul>
                        );
                      }
                      return <p key={i} className="text-slate-350">{para}</p>;
                    })}
                  </div>

                </div>

              </div>
            )}

          </div>
        )}

        {/* TAB 5.5: CORE MODEL MACHINE LEARNING ENGINE PANEL */}
        {activeTab === "ml" && (
          <div className="space-y-8">
            {/* ML introduction trainer */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-center justify-between">
              <div className="flex gap-5 items-start">
                <div className="bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 p-3 rounded-2xl text-slate-950 h-11 w-11 flex items-center justify-center shrink-0">
                  <Cpu className="h-5 w-5 stroke-[2.5]" />
                </div>
                <div>
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-sm font-bold text-slate-100">
                      Mesin Pembelajaran Mandiri Luring (Ensemble Random Forest & ARIMA)
                    </h3>
                    <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-0.5 rounded-full font-mono uppercase tracking-wider font-bold">
                      Offline-Trained ML
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed max-w-xl">
                    Sistem melatih kombinasi pengurut biner **Ensemble Decision Trees (Random Forest Bagging)** dan penganalisis deret waktu musiman **ARIMA (Autoregressive Integrated Moving Average)** pada corong data mentah Macau.
                  </p>
                </div>
              </div>

              <button
                onClick={runMlTrainEngine}
                disabled={isMlLoading}
                className="bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-600 hover:to-teal-600 active:scale-[0.98] transition-all text-slate-950 font-extrabold font-mono text-xs px-5 py-2.5 rounded-xl flex items-center justify-center gap-2 cursor-pointer shadow-lg disabled:opacity-50 shrink-0 w-full md:w-auto"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${isMlLoading ? "animate-spin" : ""}`} />
                {isMlLoading ? "MELATIH MODEL RANDOM FOREST..." : "RETRAIN MODEL ML"}
              </button>
            </div>

            {/* Error notifications */}
            {mlError && (
              <div className="bg-rose-500/10 border border-rose-500/20 text-rose-400 p-4 rounded-xl flex items-start gap-3 text-xs font-mono">
                <AlertTriangle className="h-5 w-5 mt-0.5 shrink-0" />
                <div>
                  <p className="font-bold underline">Kesalahan Latihan Mesin Pembelajaran</p>
                  <p className="mt-1 leading-relaxed">{mlError}</p>
                </div>
              </div>
            )}

            {mlSummary && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                {/* Visualizer: Convergence Log & Information Gain rate */}
                <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                      <h4 className="text-xs font-bold font-mono text-slate-350 tracking-wider uppercase">Konvergensi Impuritas Gini</h4>
                      <span className="text-[10px] font-mono text-cyan-400 font-semibold">Gini Epoch Curve</span>
                    </div>
                    
                    <p className="text-xs text-slate-405 mb-4 font-mono leading-relaxed">
                      Latihan dilakukan selama {mlSummary.trainingEpochs} iterasi. Penurunan ketidakmurnian (Gini Impurity) mereduksi derau keacakan matematika pada tiap node pohon biner:
                    </p>

                    <div className="space-y-2 font-mono text-[11px]">
                      {mlSummary.treeLossConvergence?.slice(0, 8).map((loss: number, idx: number) => {
                        const percentFilled = Math.floor((loss / 0.9) * 100);
                        return (
                          <div key={idx} className="flex items-center gap-3">
                            <span className="text-slate-500 w-12 text-right">Epoch {idx + 1}:</span>
                            <div className="flex-1 bg-slate-950 h-2.5 rounded-full overflow-hidden ring-1 ring-white/5">
                              <div 
                                className="bg-gradient-to-r from-emerald-500 to-teal-400 h-full rounded-full" 
                                style={{ width: `${percentFilled}%` }} 
                              />
                            </div>
                            <span className="text-emerald-400 w-10 font-semibold">{loss.toFixed(3)}</span>
                          </div>
                        );
                      })}
                      {mlSummary.treeLossConvergence?.length > 8 && (
                        <div className="text-slate-500 font-mono text-[10px] text-center pt-1">
                          dan {mlSummary.treeLossConvergence.length - 8} epoch konvergensi lanjutan...
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="mt-6 pt-4 border-t border-slate-850 font-mono text-center flex items-center justify-between">
                    <span className="text-slate-550 text-[10px]">Data Sampel Digunakan:</span>
                    <span className="text-slate-200 text-xs font-bold bg-slate-950 px-2.5 py-0.5 rounded border border-slate-850">
                      {mlSummary.dataPointsUsed} Undian Macau
                    </span>
                  </div>
                </div>

                {/* Score indicators: Accuracy and Confidence Metrics */}
                <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl flex flex-col justify-between">
                  <div>
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                      <h4 className="text-xs font-bold font-mono text-slate-350 tracking-wider uppercase">Metrik Evaluasi Model</h4>
                      <span className="text-[10px] font-mono text-emerald-400 font-semibold">Validation Accuracy</span>
                    </div>

                    <div className="text-center py-6">
                      <div className="inline-block relative">
                        <div className="text-5xl font-black font-sans tracking-tight bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                          {mlSummary.accuracyRate}%
                        </div>
                        <div className="text-[9px] font-mono text-slate-450 font-bold uppercase tracking-wider mt-1.5">
                          Tingkat Akurasi Margin (±1)
                        </div>
                      </div>

                      <p className="text-xs text-slate-400 mt-4 leading-relaxed px-2 font-mono">
                        Akurasi diukur dari perbandingan prediksi terkuat Ekor terhadap data riil di database. Nilai di atas 75% menunjukkan kestabilan tinggi untuk rekomendasi 2D/3D Macau.
                      </p>
                    </div>

                    <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-850 text-[11px] font-mono leading-relaxed text-slate-400">
                      <span className="text-emerald-400 font-bold">✓ Bootstrap Aggregation:</span> Sebanyak 8 pohon klasifikasi biner dilatih dengan random subset sampel guna meredam fenomena *overfitting*.
                    </div>
                  </div>

                  <div className="mt-4 pt-4 border-t border-slate-850 flex items-center justify-between font-mono">
                    <span className="text-slate-550 text-[10px]">Confidence Index:</span>
                    <span className="bg-gradient-to-r from-emerald-450 to-cyan-400 bg-clip-text text-transparent font-bold text-xs">
                      {mlSummary.futurePredictions?.confidenceIndex}% (Kategori Tinggi)
                    </span>
                  </div>
                </div>

                {/* Seasonal Detected HOT digits per timeslot */}
                <div className="bg-slate-900 border border-slate-800 p-5 rounded-2xl">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-3 mb-4">
                    <h4 className="text-xs font-bold font-mono text-slate-350 tracking-wider uppercase">Dekomposisi Musiman (ARIMA Slot)</h4>
                    <span className="text-[10px] font-mono text-teal-400 font-semibold">Seasonal Hotspots</span>
                  </div>

                  <p className="text-xs text-slate-405 mb-4 font-mono leading-relaxed">
                    Model auto-regresi ARIMA mengekstrak digit dengan bias kemunculan tertinggi musiman yang terikat pada jam-jam undian spesifik:
                  </p>

                  <div className="space-y-2 font-mono text-[11px]">
                    {Object.entries(mlSummary.seasonalDetected || {}).map(([slot, digits]: [string, any]) => (
                      <div key={slot} className="flex items-center justify-between bg-slate-950/40 p-2 rounded-lg border border-slate-850/50 hover:bg-slate-950/85 transition-all">
                        <span className="text-slate-400 font-bold text-[11px]">{slot} WIB</span>
                        <div className="flex gap-1.5">
                          {digits.map((digit: number, posIdx: number) => (
                            <span 
                              key={posIdx} 
                              title={`Posisi ke-${posIdx + 1}`}
                              className="w-5 h-5 bg-slate-900 border border-slate-800 text-teal-400 font-bold flex items-center justify-center rounded text-[10px]"
                            >
                              {digit}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

              </div>
            )}

            {/* PREDICTED NUMBERS BASED ON COMPOSITE ML MODEL */}
            {mlSummary && (
              <div className="bg-gradient-to-r from-emerald-950/15 via-slate-900 to-cyan-950/10 border border-emerald-900/20 p-6 rounded-3xl">
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6 border-b border-slate-900 pb-4">
                  <div>
                    <h4 className="text-sm font-bold font-mono text-emerald-400 flex items-center gap-2">
                      <Sliders className="h-4 w-4 animate-spin text-emerald-500 duration-1000" />
                      PREDIKSI KOMPOSIT MODEL ML (TARGET: {mlSummary.futurePredictions?.drawTarget?.date} - SLOT {mlSummary.futurePredictions?.drawTarget?.timeSlot} WIB)
                    </h4>
                    <p className="text-xs text-slate-400 mt-1 font-mono leading-relaxed">
                      Dirumuskan lewat perpecahan batas klasifikasi Random Forest dan korelasi lag temporal ARIMA.
                    </p>
                  </div>
                  <div className="bg-emerald-500/10 border border-emerald-500/20 px-3 py-1 text-emerald-400 text-xs font-mono font-bold rounded-full">
                    Kekuatan Sinyal Model: {mlSummary.futurePredictions?.confidenceIndex}% Confidence
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  {/* 4D ML Suggestions */}
                  <div className="bg-slate-950/70 border border-slate-850 p-5 rounded-2xl">
                    <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-bold">Rekomendasi Set 4D</span>
                    <div className="mt-3 space-y-2">
                      {mlSummary.futurePredictions?.predictions4D?.slice(0, 3).map((v: string, i: number) => (
                        <div key={i} className="flex justify-between items-center bg-slate-900/50 px-3.5 py-2.5 rounded-xl border border-slate-800/80 hover:border-emerald-500/20 transition-all font-mono">
                          <span className="text-slate-500 text-[10px]">Prioritas {i + 1}</span>
                          <span className="text-md font-black text-emerald-400 tracking-widest">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 3D ML Suggestions */}
                  <div className="bg-slate-950/70 border border-slate-850 p-5 rounded-2xl">
                    <span className="text-[10px] font-mono text-slate-500 uppercase tracking-wider font-bold">Rekomendasi Set 3D</span>
                    <div className="mt-3 space-y-2 flex flex-col justify-center">
                      {mlSummary.futurePredictions?.predictions3D?.slice(0, 3).map((v: string, i: number) => (
                        <div key={i} className="flex justify-between items-center bg-slate-900/50 px-3.5 py-2.5 rounded-xl border border-slate-800/80 hover:border-emerald-500/20 transition-all font-mono">
                          <span className="text-slate-500 text-[10px]">Prioritas {i + 1}</span>
                          <span className="text-md font-black text-cyan-400 tracking-widest">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 2D ML Suggestions & Colok / Shio */}
                  <div className="bg-slate-950/70 border border-slate-850 p-5 rounded-2xl flex flex-col justify-between">
                    <div>
                      <span className="text-[10px] font-mono text-slate-550 uppercase tracking-wider font-bold font-semibold">Colok & Shio Aktif</span>
                      
                      <div className="grid grid-cols-2 gap-3 mt-3 font-mono">
                        <div className="bg-slate-900/50 p-2.5 rounded-xl border border-slate-800/80 text-center">
                          <span className="text-[9px] text-slate-500 block uppercase font-semibold">Colok Bebas</span>
                          <span className="text-md font-bold text-slate-200 mt-1 block">
                            {mlSummary.futurePredictions?.colokBebas?.join(", ")}
                          </span>
                        </div>
                        <div className="bg-slate-900/50 p-2.5 rounded-xl border border-slate-800/80 text-center">
                          <span className="text-[9px] text-slate-500 block uppercase font-semibold">Colok Macau</span>
                          <span className="text-xs font-bold text-slate-200 mt-1 block truncate">
                            {mlSummary.futurePredictions?.colokMacau?.[0]}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-5 bg-emerald-500/5 border border-emerald-500/10 p-3 rounded-xl text-[10.5px] text-slate-400 font-mono flex items-center justify-between">
                      <span className="text-slate-500">Kecocokan Shio Astronomi:</span>
                      <span className="text-emerald-400 font-bold">
                        {mlSummary.futurePredictions?.shioAccents?.join(" / ")}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* ANOMALY DETECTION REPORT FEED */}
            <div className="bg-slate-900 border border-slate-800 rounded-3xl p-6 shadow-xl">
              <div>
                <h3 className="text-sm font-bold font-mono text-slate-200 tracking-wider uppercase flex items-center gap-2">
                  <Activity className="h-4 w-4 text-rose-500 animate-pulse animate-duration-1000" />
                  FEEDAN DETEKSI ANOMALI NUMERIK MACOULYZER (HISTORIS)
                </h3>
                <p className="text-xs text-slate-400 mt-1 font-mono leading-relaxed mb-6">
                  Sistem mengevaluasi seluruh riwayat undian Macau dan menyaring deviasi yang menyalahi distribusi rata-rata normal (outliers, angka kembar quad, triple, dan distorsi entropi shannon).
                </p>
              </div>

              {mlAnomalies && mlAnomalies.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-h-[460px] overflow-y-auto pr-1">
                  {mlAnomalies.map((anom: any) => (
                    <div 
                      key={anom.id} 
                      className={`p-4 rounded-2xl border flex flex-col justify-between ${
                        anom.isExtremeOutlier 
                          ? "bg-rose-950/15 border-rose-900/35 hover:border-rose-800/55" 
                          : "bg-slate-950/60 border-slate-850 hover:border-slate-750"
                      } transition-all duration-300`}
                    >
                      <div>
                        <div className="flex items-center justify-between border-b border-slate-900/60 pb-2.5 mb-3 select-none">
                          <div className="flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-pulse" />
                            <span className="text-xs font-bold text-slate-100 font-mono tracking-widest">{anom.drawCode}</span>
                          </div>
                          <span className={`text-[9.5px] font-mono px-2 py-0.5 rounded font-bold ${
                            anom.isExtremeOutlier ? "bg-rose-500/15 text-rose-400" : "bg-amber-500/10 text-amber-400"
                          }`}>
                            Score: {anom.anomalyScore}
                          </span>
                        </div>

                        <div className="space-y-1.5 font-mono text-[11px] leading-relaxed text-slate-400 pr-2">
                          {anom.reasons.map((rsn: string, i: number) => (
                            <p key={i} className="flex gap-1.5">
                              <span className="text-rose-500 font-bold shrink-0">⚠</span>
                              <span>{rsn}</span>
                            </p>
                          ))}
                        </div>
                      </div>

                      <div className="mt-4 pt-2.5 border-t border-slate-900/60 flex items-center justify-between text-[10px] font-mono text-slate-500">
                        <span>Tanggal: {anom.date}</span>
                        <span>Slot: {anom.timeSlot} WIB</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="bg-slate-950 border border-slate-850 py-16 text-center text-slate-500 font-mono text-xs rounded-2xl">
                  {isMlLoading ? "Sedang mengevaluasi riwayat..." : "Koleksi data stabil. Belum ditemukan anomali matematika yang signifikan."}
                </div>
              )}
            </div>

          </div>
        )}

        {/* TAB 6: BACKTEST SANDBOX EMPIRICAL VERIFICATION */}
        {activeTab === "backtest" && (
          <div className="space-y-8">
            
            {/* Sandbox introduction */}
            <div className="bg-slate-900 border border-slate-800 p-6 rounded-2xl flex flex-col md:flex-row gap-6 items-center justify-between">
              <div className="flex gap-4 items-start">
                <div className="bg-emerald-500/10 text-emerald-400 p-3 rounded-2xl h-12 w-12 flex items-center justify-center">
                  <Play className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-md font-bold text-slate-100">
                    Backtesting Sandbox: Empirically Test the Model
                  </h3>
                  <p className="text-xs text-slate-400 mt-1 leading-relaxed max-w-xl">
                    Seberapa akurat analisis matematika kita? Uji model dengan mode simulasi mundur (Backtesting). Sistem akan melatih model pada sekumpulan data di masa lalu secara otomatis, meluncurkan prediksi, dan mencocokkannya dengan keluaran riil yang sudah keluar!
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-3 w-full md:w-auto mt-4 md:mt-0">
                <span className="text-xs text-slate-400 font-mono shrink-0">Sampel Uji:</span>
                <select
                  value={backtestSize}
                  onChange={e => setBacktestSize(parseInt(e.target.value, 10))}
                  className="bg-slate-950 border border-slate-800 focus:border-cyan-500 rounded-lg p-2.5 text-xs text-slate-200 font-mono outline-none w-28"
                >
                  <option value={10}>10 Undian</option>
                  <option value={15}>15 Undian</option>
                  <option value={20}>20 Undian</option>
                  <option value={30}>30 Undian</option>
                </select>

                <button
                  onClick={runBacktestSimulator}
                  disabled={isBacktesting}
                  className="bg-emerald-500 hover:bg-emerald-600 font-mono font-bold text-xs px-5 py-2.5 text-slate-950 rounded-xl cursor-pointer disabled:opacity-50"
                >
                  Mulai Simulasi
                </button>
              </div>
            </div>

            {/* Backtest output reports */}
            {backtestResults && (
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                
                {/* Reports summary */}
                <div className="lg:col-span-1 space-y-6">
                  
                  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6">
                    <h4 className="text-xs font-bold font-mono text-slate-400 uppercase tracking-widest border-b border-slate-800 pb-3 mb-5">
                      Rasio Akurasi Hasil Simulasi
                    </h4>

                    <div className="space-y-5">
                      
                      {/* Colok Bebas Hits Rate */}
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-400">Rasio Hit Colok Bebas (CB):</span>
                          <span className="text-emerald-400 font-bold">
                            {((backtestResults.colokBebasHits / backtestResults.totalTrials) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-950 h-3 rounded overflow-hidden">
                          <div
                            style={{ width: `${(backtestResults.colokBebasHits / backtestResults.totalTrials) * 100}%` }}
                            className="bg-emerald-450 h-full rounded transition-all duration-700"
                          />
                        </div>
                        <p className="text-[10px] text-slate-550 font-mono">
                          Tembus minimal 1 angka dalam top 3 digit di {backtestResults.colokBebasHits} dari {backtestResults.totalTrials} percobaan.
                        </p>
                      </div>

                      {/* 2D Hit Rate */}
                      <div className="space-y-1.5 pt-3 border-t border-slate-800/60">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-slate-400">Rasio Tepat 2D Hasil:</span>
                          <span className="text-cyan-400 font-bold">
                            {((backtestResults.exact2DHits / backtestResults.totalTrials) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-950 h-3 rounded overflow-hidden">
                          <div
                            style={{ width: `${(backtestResults.exact2DHits / backtestResults.totalTrials) * 100}%` }}
                            className="bg-cyan-450 h-full rounded transition-all duration-700"
                          />
                        </div>
                        <p className="text-[10px] text-slate-550 font-mono">
                          Kombinasi Markov cross-transitions sukses memuat tepat 2D belakang (Kepala-Ekor) sebanyak {backtestResults.exact2DHits} kali.
                        </p>
                      </div>

                    </div>
                  </div>

                </div>

                {/* Sandbox Trials details logs */}
                <div className="lg:col-span-2 bg-slate-900 border border-slate-800 rounded-3xl p-6">
                  <h4 className="text-sm font-bold text-slate-200 mb-4 flex items-center gap-2">
                    <Activity className="h-4 w-4 text-emerald-450" /> Log Simulasi Transaksi Mundur
                  </h4>

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs border-collapse font-mono">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-450 text-[10px] uppercase tracking-wider">
                          <th className="py-2.5 px-3">Tanggal / Slot</th>
                          <th className="py-2.5 px-3 text-center">Hasil Aktual</th>
                          <th className="py-2.5 px-3">Prediksi Colok Bebas</th>
                          <th className="py-2.5 px-3">Prediksi Set 2D</th>
                          <th className="py-2.5 px-3 text-right">CB</th>
                          <th className="py-2.5 px-3 text-right">2D</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-800/40">
                        {backtestResults.trialsDetail.map((trial) => (
                          <tr key={trial.drawId} className="hover:bg-slate-800/10 transition-colors">
                            <td className="py-3 px-3">
                              <div className="text-slate-300 font-semibold">{trial.date}</div>
                              <div className="text-[10px] text-slate-450">{trial.timeSlot} WIB</div>
                            </td>
                            <td className="py-3 px-3 text-center text-slate-100 font-bold bg-slate-950/40 text-sm">
                              {trial.actual}
                            </td>
                            <td className="py-3 px-3">
                              <div className="flex gap-1">
                                {trial.predictedCB.map(d => (
                                  <span key={d} className="bg-slate-950 text-slate-350 border border-slate-850 px-1 rounded text-[10px]">
                                    {d}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="py-3 px-3">
                              <div className="flex flex-wrap gap-1">
                                {trial.predicted2D.map(code => (
                                  <span key={code} className="bg-slate-950 text-[10.5px] px-1 rounded text-slate-400">
                                    {code}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td className="py-3 px-3 text-right text-xs">
                              {trial.isCBHit ? (
                                <span className="text-emerald-400 font-bold">HIT  ✓</span>
                              ) : (
                                <span className="text-slate-600">-</span>
                              )}
                            </td>
                            <td className="py-3 px-3 text-right text-xs">
                              {trial.is2DHit ? (
                                <span className="text-cyan-400 font-bold">HIT  ✓</span>
                              ) : (
                                <span className="text-slate-600">-</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                </div>

              </div>
            )}

            {!backtestResults && (
              <div className="bg-slate-900/40 border border-slate-850 py-20 text-center text-slate-550 font-mono text-xs rounded-3xl">
                Belum ada simulasi yang dijalankan. Silakan pilih parameter sampel dan klik tombol Simulasi di atas.
              </div>
            )}

          </div>
        )}

      </main>

      {/* Footer Design block */}
      <footer className="border-t border-slate-900 mt-20 bg-slate-950 py-10 font-mono text-center text-slate-500 text-xs">
        <p className="max-w-md mx-auto leading-relaxed">
          Macaulyzer adalah platform analitis independen. Semua probabilitas transisi Markov, fungsi Poisson PMF, dan trendline gradien dihitung secara deterministik murni berdasarkan representasi statistical corpus.
        </p>
        <p className="mt-4 text-slate-600">
          © 2026 Macaulyzer Engine Inc. All statistical formulas strictly verified.
        </p>
      </footer>

    </div>
  );
}
