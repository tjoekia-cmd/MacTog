/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export interface MacauDraw {
  id: string;
  date: string;       // YYYY-MM-DD
  timeSlot: string;   // e.g., "13:00", "16:00", "19:00", "22:00", "00:01"
  drawCodeString: string; // e.g., "5432"
  digits: [number, number, number, number]; // [As, Kop, Kepala, Ekor]
  sum: number;
  isFromOcr?: boolean;
}

export interface StatisticalMetrics {
  totalSample: number;
  frequencies: {
    as: Record<number, number>;
    kop: Record<number, number>;
    kepala: Record<number, number>;
    ekor: Record<number, number>;
    overall: Record<number, number>;
  };
  hotNumbers: { digit: number; count: number }[];
  coldNumbers: { digit: number; count: number }[];
  evenOddRatio: { even: number; odd: number };
  bigSmallRatio: { big: number; small: number }; // 0-4 Small, 5-9 Big
}

export interface PoissonPrediction {
  position: "As" | "Kop" | "Kepala" | "Ekor";
  probabilities: { digit: number; probability: number }[]; // Poisson PMF values
  mostProbable: number;
}

export interface MarkovTransitionMatrix {
  matrix: number[][]; // 10x10 transition matrix for a specific position
  currentState: number;
  nextStateProbabilities: { digit: number; probability: number }[];
}

export interface RegressionTrend {
  slope: number;
  intercept: number;
  predictedNextSum: number;
  rSquared: number;
  gradientDescentSteps: { step: number; loss: number; w: number; b: number }[];
}

export interface ChiSquareResult {
  chiSquareStat: number;
  criticalValue: number;
  pValue: number;
  isBiased: boolean; // True if p-value < 0.05, meaning statistical bias exists
  entropy: number;   // Shannon Entropy measuring randomness chaos
}

export interface PredictionEngineOutput {
  drawTarget: { date: string; timeSlot: string };
  poisson: PoissonPrediction[];
  markov: {
    as: MarkovTransitionMatrix;
    kop: MarkovTransitionMatrix;
    kepala: MarkovTransitionMatrix;
    ekor: MarkovTransitionMatrix;
  };
  regression: RegressionTrend;
  chiSquare: ChiSquareResult;
  suggestedNumbers: {
    numbers4D: string[];
    numbers3D: string[];
    numbers2D: string[];
    colokBebas: number[];
    colokMacau: string[];
    confidence: number;
  };
}

export interface AiPredictionResponse {
  drawTarget: { date: string; timeSlot: string };
  confidenceIndex: number;
  aiJustification: string; // Markdown text of the math, calculus, and stats reasoning
  predictions4D: string[];
  predictions3D: string[];
  predictions2D: string[];
  colokBebas: number[];
  colokMacau: string[];
  shioAccents: string[];
  mathematicalFormulasUsed: string[];
}
