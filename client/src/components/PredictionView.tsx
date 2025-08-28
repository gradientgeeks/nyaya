import React from 'react';
import type { CasePrediction } from '../types';

interface PredictionViewProps {
  data: CasePrediction;
}

export const PredictionView: React.FC<PredictionViewProps> = ({ data }) => {
  return (
    <div className="space-y-4 sm:space-y-6">
      <h3 className="font-bold text-base sm:text-lg mb-3 text-gray-800 dark:text-gray-200">
        Probable Judgment Prediction
      </h3>
      <div className="space-y-3 mb-6">
        {data.probabilities.map((p, i) => (
          <div key={i} className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
            <span className="flex-1 text-sm sm:text-base text-gray-700 dark:text-gray-300 font-medium">
              {p.outcome}
            </span>
            <div className="flex items-center gap-3 flex-1">
              <div className="flex-1 bg-gray-200 dark:bg-gray-700 rounded-full h-2.5 min-w-0">
                <div 
                  className="bg-blue-500 h-2.5 rounded-full transition-all duration-500" 
                  style={{ width: p.probability }}
                ></div>
              </div>
              <span className="font-semibold text-sm text-blue-600 dark:text-blue-400 min-w-max">
                {p.probability}
              </span>
            </div>
          </div>
        ))}
      </div>
      
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <h4 className="font-semibold text-sm sm:text-base mb-3 text-gray-800 dark:text-gray-200">
          Based on Precedents:
        </h4>
        <ul className="space-y-2">
          {data.precedents.map((p, i) => (
            <li key={i} className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-2 flex-shrink-0"></span>
              <span className="text-xs sm:text-sm text-gray-600 dark:text-gray-400 leading-relaxed">
                {p}
              </span>
            </li>
          ))}
        </ul>
      </div>
      
      <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-lg p-3 mt-4">
        <p className="text-xs text-yellow-800 dark:text-yellow-200 italic">
          <strong>Disclaimer:</strong> This is a predictive analysis based on historical data and should not be considered legal advice.
        </p>
      </div>
    </div>
  );
};