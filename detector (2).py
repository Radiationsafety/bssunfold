"""Detector class with unfolding methods."""

import os
import sys



import numpy as np
from typing import Dict, Optional, List, Tuple, Any
from scipy.optimize import minimize
import cvxpy as cp
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

import seaborn as sns
import odl
from cvxopt import matrix, solvers

from cuqi.model import LinearModel
from cuqi.distribution import Gaussian, LMRF, CMRF, Gamma
from cuqi.problem import BayesianProblem

from scipy.optimize import minimize
from scipy.stats import norm
from scipy.linalg import svd
from scipy.sparse.linalg import (
    bicgstab,
    cg,
    cgs,
    gmres,
    lgmres,
    minres,
    qmr,
    gcrotmk,
    tfqmr,
    lsmr,
    lsqr,
)

import gurobipy as gp
from gurobipy import GRB

import pyomo.environ as pyo
from pyomo.environ import SolverFactory

import lmfit

# import emcee

import random
from deap import base, creator, tools, algorithms

from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
from pyunfold.callbacks import SplineRegularizer

import kaczmarz

from statreg.model import GaussErrorMatrixUnfolder
from statreg.basis import CubicSplines, FourierBasis

import pymc as pm

# import arviz as az

from qunfold import QUnfolder, QPlotter
from qunfold.utils import normalize_response, lambda_optimizer


# from rf_functions import convert_rf_to_matrix_variable_step

# External dependencies (assuming they exist as shown in example usage)
import warnings

try:
    from spectrum_recovery import calculate_spectrum_tikhonov_core
    from maxed import calculate_spectrum_maxed_core
    from gravel import gravel, gravel_ball_system_with_errors
    from constants import ICRP116_COEFF_EFFECTIVE_DOSE
except ImportError:
    warnings.warn("External modules not found. Some methods will be unavailable.")


class Detector:
    """
    Класс для работы с детекторами и восстановления спектра нейтронов.
    """

    # def __init__(self, E_MeV: np.ndarray, sensitivities: Dict[str, np.ndarray]):
    #     """
    #     Инициализация детектора.

    #     Параметры:
    #     E_MeV : np.ndarray
    #         Энергетические бины [МэВ]
    #     sensitivities : dict[str, np.ndarray]
    #         {имя_детектора: массив_чувствительности}, длина = len(E_MeV)
    #     """
    #     self.E_MeV = np.asarray(E_MeV, dtype=float)
    #     if self.E_MeV.ndim != 1:
    #         raise ValueError("E_MeV должен быть одномерным массивом")
    #     if len(self.E_MeV) < 2:
    #         raise ValueError("Требуется минимум 2 энергетических бина")

    #     if not sensitivities:
    #         raise ValueError("Не указаны функции чувствительности детекторов")

    #     self.detector_names = list(sensitivities.keys())
    #     self.sensitivities = {}
    #     for name, sens in sensitivities.items():
    #         sens = np.asarray(sens, dtype=float)
    #         if sens.shape != self.E_MeV.shape:
    #             raise ValueError(
    #                 f"Длина чувствительности '{name}' не совпадает с E_MeV"
    #             )
    #         self.sensitivities[name] = sens

    #     self.cc_icrp116 = self._load_icrp116_coefficients()  # для расчёта дозы
    #     # self.Amat = self._convert_rf_to_matrix_variable_step()  # делаем расчётную матрицу из dataframe

    def __init__(self, response_functions_df):

        Amat, E_MeV, detector_names, log_steps = (
            self._convert_rf_to_matrix_variable_step(response_functions_df, Emin=1e-9)
        )

        self.Amat = Amat
        self.E_MeV = np.asarray(E_MeV, dtype=float)
        self.detector_names = detector_names
        self.log_steps = log_steps

        if self.E_MeV.ndim != 1:
            raise ValueError("E_MeV должен быть одномерным массивом")
        if len(self.E_MeV) < 2:
            raise ValueError("Требуется минимум 2 энергетических бина")

        self.sensitivities = {
            self.detector_names[i]: np.array(Amat[:, i])
            for i in range(len(self.detector_names))
        }
        self.cc_icrp116 = self._load_icrp116_coefficients()  # для расчёта дозы

    def __str__(self) -> str:
        """
        Строковое представление детектора для пользователя.

        Возвращает:
        str
            Информация о детекторе в читаемом формате.
        """
        energy_range = f"{self.E_MeV[0]:.3e} - {self.E_MeV[-1]:.3e} МэВ"
        return (
            f"Detector(энергетических бинов: {self.n_energy_bins}, "
            f"детекторов: {self.n_detectors}, "
            f"диапазон: {energy_range})"
        )

    def __repr__(self) -> str:
        """
        Техническое строковое представление детектора.

        Возвращает:
        str
            Строка, которую можно использовать для воссоздания объекта.
        """
        return (
            f"Detector(E_MeV={self.E_MeV.tolist()}, sensitivities={self.sensitivities})"
        )

    @property
    def n_detectors(self) -> int:
        """Количество доступных детекторов."""
        return len(self.detector_names)

    @property
    def n_energy_bins(self) -> int:
        """Количество энергетических бинов."""
        return len(self.E_MeV)

    def _validate_readings(self, readings: Dict[str, float]) -> Dict[str, float]:
        """
        Проверка и валидация показаний детекторов.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов

        Возвращает:
        Dict[str, float]
            Проверенные показания

        Исключения:
        ValueError
            Если показания отрицательные или не указаны показания ни одного детектора
        """
        valid = {}
        for det in self.detector_names:
            if det in readings:
                val = float(readings[det])
                if val < 0:
                    raise ValueError(f"Показание '{det}' отрицательное: {val}")
                valid[det] = val
        if not valid:
            raise ValueError("Не указаны показания ни одного детектора")
        return valid

    def _build_system(
        self, readings: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Build response matrix A and measurement vector b."""
        selected = [name for name in self.detector_names if name in readings]
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)
        return A, b, selected

    # TODO: сделать единый выход по всем методам    
    def _standardize_output(
        self,
        spectrum: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        selected: List[str],
        method: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create standardized output dictionary for all methods."""
        # Гарантируем неотрицательность спектра
        spectrum_nonneg = np.maximum(spectrum, 0)
        
        computed_readings = A @ spectrum_nonneg
        residual = b - computed_readings
        
        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": spectrum_nonneg.copy(),  # Используем неотрицательную версию
            "spectrum_absolute": spectrum_nonneg.copy(),  # Тоже неотрицательная
            "effective_readings": {
                name: float(val) for name, val in zip(selected, computed_readings)
            },
            "residual": residual.copy(),
            "residual norm": float(np.linalg.norm(residual)),
            "method": method,
            "doserates": self._calculate_doserates(spectrum_nonneg),
        }
        output.update(kwargs)
        return output
    
    def _convert_rf_to_matrix_variable_step(self, rf_df, Emin=1e-9) -> np.ndarray:
        """
        Преобразует DataFrame с функциями чувствительности в матрицу,
        умножая на np.log(10) и на индивидуальный шаг по логарифмической энергии
        для каждой точки.

        Parameters:
        -----------
        rf_df : pd.DataFrame
            DataFrame с функциями чувствительности.
            Первая колонка 'E_MeV' - энергии в МэВ.
            Остальные колонки - функции чувствительности для разных сфер.

        Returns:
        --------
        tuple: (matrix, energies, sphere_names, log_steps)
            matrix : np.ndarray
                Матрица размером (n_energies, n_spheres)
            energies : np.ndarray
                Массив энергий в МэВ
            sphere_names : list
                Список имен сфер
            log_steps : np.ndarray
                Массив шагов в логарифмической шкале для каждой точки
        """

        # Извлекаем энергии
        if "E_MeV" in rf_df.columns:
            energies = rf_df["E_MeV"].values
            rf_data = rf_df.drop("E_MeV", axis=1)
        else:
            # Предполагаем, что первый столбец - это энергия
            energies = rf_df.iloc[:, 0].values
            rf_data = rf_df.iloc[:, 1:]

        # Получаем имена сфер
        sphere_names = rf_data.columns.tolist()

        # Преобразуем в numpy массив
        rf_array = rf_data.values  # размер: (n_energies, n_spheres)

        # Вычисляем логарифмы энергий
        log_energies = np.log10(energies / Emin)

        # Вычисляем шаги в логарифмической шкале для каждой точки
        # Для внутренних точек используем среднее шагов слева и справа
        # Для граничных точек используем односторонний шаг

        n_points = len(energies)
        log_steps = np.zeros(n_points)

        # Для первой точки
        log_steps[0] = log_energies[1] - log_energies[0]

        # Для последней точки
        log_steps[-1] = log_energies[-1] - log_energies[-2]

        # Для внутренних точек
        for i in range(1, n_points - 1):
            # Средний шаг между левым и правым интервалами
            left_step = log_energies[i] - log_energies[i - 1]
            right_step = log_energies[i + 1] - log_energies[i]
            log_steps[i] = (left_step + right_step) / 2

        # Из-за перехода к логарифмической шкале,  нужно домножить на np.log(10)

        ln_steps = log_steps * np.log(10)

        # Умножаем каждую строку матрицы на соответствующий шаг
        # Используем broadcasting: (n_energies, n_spheres) * (n_energies, 1)
        rf_matrix = rf_array * ln_steps[:, np.newaxis]

        return rf_matrix, energies, sphere_names, log_steps

    def unfold_tikhonov_legendre(
        self, readings: Dict[str, float], delta: float = 0.05, n_polynomials: int = 15
    ) -> Dict:
        """
        Восстановление спектра методом регуляризации Тихонова.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        delta : float, optional
            Параметр регуляризации. По умолчанию 0.05
        n_polynomials : int, optional
            Количество полиномов Лежандра для разложения. По умолчанию 15

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        K_j = np.array([self.sensitivities[name] for name in selected], dtype=float)

        result = calculate_spectrum_tikhonov_core(
            E_MeV=self.E_MeV, K_j=K_j, Q_j=Q_j, delta=delta, n_polynomials=n_polynomials
        )
        result["effective_readings"] = {
            name: float(val)
            for name, val in zip(selected, result["effective_readings"])
        }
        # невязка и норма невязки
        residual = K_j @ result["spectrum"] - Q_j
        result["residual"] = residual
        result["residual norm"] = np.linalg.norm(residual)
        result["doserates"] = self._calculate_doserates(result["spectrum"])
        return result

    def unfold_maxed(
        self,
        readings: Dict[str, float],
        reference_spectrum: Optional[Dict[str, np.ndarray]] = None,
        sigma_factor: float = 0.01,
        omega: float = 1.0,
        maxiter: int = 5000,
        tol: float = 1e-6,
    ) -> Dict:
        """
        Восстановление спектра методом MAXED (Maximum Entropy Deconvolution).

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        reference_spectrum : Optional[Dict[str, np.ndarray]], optional
            Опорный спектр. Если None, используется спектр по умолчанию.
        sigma_factor : float, optional
            Фактор погрешности измерений. По умолчанию 0.01
        omega : float, optional
            Параметр ограничения хи-квадрат. По умолчанию 1.0
        maxiter : int, optional
            Максимальное число итераций. По умолчанию 5000
        tol : float, optional
            Точность оптимизации. По умолчанию 1e-6

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        K_j = np.array([self.sensitivities[name] for name in selected], dtype=float)

        phi_0 = None
        if reference_spectrum is not None:
            ref_E = np.asarray(reference_spectrum["E_MeV"], dtype=float)
            ref_phi = np.asarray(reference_spectrum["Phi"], dtype=float)
            if len(ref_E) != len(ref_phi):
                raise ValueError("Длины E_MeV и Phi в reference_spectrum не совпадают")
            phi_0 = np.interp(
                np.log10(self.E_MeV),
                np.log10(np.maximum(ref_E, self.E_MeV.min())),
                ref_phi,
                left=1e-10,
                right=1e-10,
            )
            phi_0 = np.maximum(phi_0, 1e-10)

        result = calculate_spectrum_maxed_core(
            E_MeV=self.E_MeV,
            K_j=K_j,
            Q_j=Q_j,
            phi_0=phi_0,
            sigma_factor=sigma_factor,
            omega=omega,
            maxiter=maxiter,
            tol=tol,
        )

        residual = K_j @ result["spectrum"] - Q_j
        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": result["spectrum"],
            "spectrum_absolute": result["spectrum"],
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, result["effective_readings"])
            },
            "residual": residual,
            "residual norm": np.linalg.norm(residual),
            "omega": result["omega"],
            "mu": str(result["mu"]),
            "method": "maxed",
            "doserates": self._calculate_doserates(result["spectrum"]),
        }

        return output

    def unfold_gravel(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма GRAVEL.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-8
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        K_j = np.array([self.sensitivities[name] for name in selected], dtype=float)
        S = K_j
        if initial_spectrum is None:
            x0 = np.ones(self.n_energy_bins)
        else:
            x0 = np.asarray(initial_spectrum, dtype=float)
            if len(x0) != self.n_energy_bins:
                raise ValueError(
                    f"Длина initial_spectrum ({len(x0)}) не совпадает с количеством энергетических бинов ({self.n_energy_bins})"
                )
        if calculate_errors:
            results = gravel_ball_system_with_errors(
                S=S,
                measurements=Q_j,
                x0=x0,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
                calculate_errors=calculate_errors,
            )
        else:
            results = gravel(
                S=S,
                measurements=Q_j,
                x0=x0,
                tolerance=tolerance,
                max_iterations=max_iterations,
                regularization=regularization,
            )

        residual = results["computed_measurements"] - Q_j
        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": results["spectrum_absolute"],
            "spectrum_absolute": results["spectrum_absolute"],
            "effective_readings": {
                name: float(val)
                for name, val in zip(selected, results["computed_measurements"])
            },
            "residual": residual,
            "residual norm": np.linalg.norm(residual),
            "iterations": results["iterations"],
            "converged": str(results["converged"]),
            "method": "gravel",
            "doserates": self._calculate_doserates(results["spectrum_absolute"]),
        }
        if "error_history" in results:
            output["error_history"] = results["error_history"]
        if "chi_sq_history" in results:
            output["chi_sq_history"] = results["chi_sq_history"]
        if "spectrum_errors" in results:
            output["spectrum_errors"] = results["spectrum_errors"]
        if "covariance_matrix" in results:
            output["covariance_matrix"] = results["covariance_matrix"]
        if "correlation_matrix" in results:
            output["correlation_matrix"] = results["correlation_matrix"]

        return output

    def unfold_tikhonov_legendre_maxed(
        self,
        readings: Dict[str, float],
        delta: float = 0.05,
        n_polynomials: int = 15,
        sigma_factor: float = 0.01,
        omega: float = 1.0,
        maxiter: int = 5000,
        tol: float = 1e-6,
    ) -> Dict:
        """
        Гибридный метод: сначала Тихонов, затем MAXED.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        delta : float, optional
            Параметр регуляризации Тихонова. По умолчанию 0.05
        n_polynomials : int, optional
            Количество полиномов Лежандра для Тихонова. По умолчанию 15
        sigma_factor : float, optional
            Фактор погрешности для MAXED. По умолчанию 0.01
        omega : float, optional
            Параметр ограничения хи-квадрат для MAXED. По умолчанию 1.0
        maxiter : int, optional
            Максимальное число итераций для MAXED. По умолчанию 5000
        tol : float, optional
            Точность оптимизации для MAXED. По умолчанию 1e-6

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        K_j = np.array([self.sensitivities[name] for name in selected], dtype=float)
        S = K_j

        tikh = self.unfold_tikhonov_legendre(readings, delta, n_polynomials)
        prior = {"E_MeV": tikh["energy"], "Phi": tikh["spectrum"]}
        maxed = self.unfold_maxed(
            readings,
            reference_spectrum=prior,
            sigma_factor=sigma_factor,
            omega=omega,
            maxiter=maxiter,
            tol=tol,
        )
        maxed["tikhonov_alpha"] = tikh.get("alpha")
        maxed["tikhonov_coefficients"] = tikh.get("coefficients")

        residual = S @ maxed["spectrum"] - Q_j
        maxed["residual"] = residual
        maxed["residual norm"] = np.linalg.norm(residual)
        maxed["doserates"] = self._calculate_doserates(maxed["spectrum"])

        return maxed

    def unfold_tikhonov_legendre_gravel(
        self,
        readings: Dict[str, float],
        delta: float = 0.05,
        n_polynomials: int = 15,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Гибридный метод: сначала Тихонов, затем GRAVEL.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        delta : float, optional
            Параметр регуляризации Тихонова. По умолчанию 0.05
        n_polynomials : int, optional
            Количество полиномов Лежандра для Тихонова. По умолчанию 15
        tolerance : float, optional
            Точность сходимости для GRAVEL. По умолчанию 1e-6
        max_iterations : int, optional
            Максимальное число итераций для GRAVEL. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации для GRAVEL. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок для GRAVEL. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        R = np.array([self.sensitivities[name] for name in selected], dtype=float)

        tikh = self.unfold_tikhonov_legendre(readings, delta, n_polynomials)
        gravel_result = self.unfold_gravel(
            readings,
            initial_spectrum=tikh["spectrum"],
            tolerance=tolerance,
            max_iterations=max_iterations,
            regularization=regularization,
            calculate_errors=calculate_errors,
        )
        gravel_result["tikhonov_alpha"] = tikh.get("alpha")
        gravel_result["tikhonov_coefficients"] = tikh.get("coefficients")
        gravel_result["method"] = "tikhonov_then_gravel"
        residual = R @ gravel_result["spectrum"] - Q_j
        gravel_result["residual"] = residual
        gravel_result["doserates"] = self._calculate_doserates(
            gravel_result["spectrum"]
        )
        gravel_result["residual norm"] = np.linalg.norm(residual)
        return gravel_result

    def unfold_mlem(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        max_iterations: int = 1000,
        tolerance: float = 1e-8,
    ) -> Dict:
        """
        Восстановление спектра методом максимального правдоподобия (MLEM).

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        method : str, optional
            Метод оптимизации ('L-BFGS-B', 'SLSQP', etc.). По умолчанию 'L-BFGS-B'.
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000.
        tolerance : float, optional
            Точность оптимизации. По умолчанию 1e-8.

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        Q_j = np.array([readings[name] for name in selected], dtype=float)
        b = Q_j
        R = np.array([self.sensitivities[name] for name in selected], dtype=float)
        if initial_spectrum is None:
            spectrum_initial = np.ones(self.n_energy_bins) * np.mean(Q_j) / np.mean(R)
        else:
            spectrum_initial = np.asarray(initial_spectrum, dtype=float)
            if len(spectrum_initial) != self.n_energy_bins:
                raise ValueError(
                    f"Длина initial_spectrum ({len(initial_spectrum)}) не совпадает с количеством энергетических бинов ({self.n_energy_bins})"
                )

        def negative_log_likelihood(spectrum, R, Q):
            """
            Функция отрицательного логарифма правдоподобия для восстановления спектра
            """
            Q_pred = np.dot(R, spectrum)
            epsilon = 1e-10
            log_likelihood = np.sum(Q * np.log(Q_pred + epsilon) - Q_pred)

            return -log_likelihood

        def gradient_negative_log_likelihood(spectrum, R, Q):
            """
            Градиент функции отрицательного логатурма правдоподобия
            """
            Q_pred = np.dot(R, spectrum)
            grad = -np.dot(R.T, (Q / (Q_pred + 1e-10) - 1))
            return grad

        bounds = [(1e-10, None)] * self.n_energy_bins
        result = minimize(
            negative_log_likelihood,
            spectrum_initial,
            args=(R, Q_j),
            method=method,
            bounds=bounds,
            jac=gradient_negative_log_likelihood,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )
        computed_readings = R @ result.x

        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": result.x,
            "spectrum_absolute": result.x,
            "doserates": self._calculate_doserates(result.x),
            "effective_readings": {
                name: float(val) for name, val in zip(selected, computed_readings)
            },
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": result.nit,
            "converged": str(result.success),
            "method": "mlem",
            "optimization_result": {
                "fun": result.fun,
                "message": result.message,
                "nfev": result.nfev,
                "njev": result.njev,
            },
        }

        return output

    # -----------------------------------------------------------

    def unfold_doroshenko(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма GRAVEL.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-6
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Инициализация спектра
        if initial_spectrum is None:
            x = np.ones(self.n_energy_bins)
        else:
            x = np.asarray(initial_spectrum, dtype=float)
            if x.size != self.n_energy_bins:
                raise ValueError(
                    f"Длина initial_spectrum ({x.size}) не совпадает с "
                    f"количеством энергетических бинов ({self.n_energy_bins})"
                )

        # Предварительные вычисления для оптимизации
        denominator_cache = np.sum(A * A, axis=0)  # ∑ A[j,i]² для каждого i

        # Основной цикл итераций
        for iteration in range(max_iterations):
            x_old = x.copy()

            # Векторизованное обновление
            for i in range(x.size):
                # Вычисление Ax без i-го элемента
                Ax_without_i = A[:, :i] @ x[:i] + A[:, i + 1 :] @ x[i + 1 :]

                # Числитель: ∑ A[j,i] * (b[j] - ∑ A[j,k] * x[k] для k≠i)
                numerator = np.dot(A[:, i], b - Ax_without_i)

                # Обновление x[i] с регуляризацией
                denominator = denominator_cache[i] + regularization
                if denominator > 0:
                    x[i] = max(0.0, numerator / denominator)

            # Проверка сходимости
            if np.linalg.norm(x - x_old) < tolerance:
                break

        # Формирование результатов
        computed_readings = A @ x

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "doserates": self._calculate_doserates(x),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": iteration + 1,
            "method": "doroshenko",
        }

    def unfold_doroshenko_matrix(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма GRAVEL.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-8
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]
        Q_j = np.array([readings[name] for name in selected], dtype=float)
        K_j = np.array([self.sensitivities[name] for name in selected], dtype=float)
        S = K_j
        if initial_spectrum is None:
            x = np.ones(self.n_energy_bins)
        else:
            x = np.asarray(initial_spectrum, dtype=float)
            if len(x) != self.n_energy_bins:
                raise ValueError(
                    f"Длина initial_spectrum ({len(x)}) не совпадает с количеством энергетических бинов ({self.n_energy_bins})"
                )

        m, n = S.shape
        A = S
        b = Q_j

        ATA = A.T @ A
        ATb = A.T @ b

        for iteration in range(max_iterations):
            x_old = x.copy()

            for i in range(n):
                # Вычисляем сумму без i-го элемента
                sum_without_i = np.sum(ATA[i, :] * x) - ATA[i, i] * x[i]

                numerator = ATb[i] - sum_without_i

                if ATA[i, i] > 0:
                    x[i] = max(0, numerator / ATA[i, i])

            if np.linalg.norm(x - x_old) < tolerance:
                print(f"Сходимость достигнута на итерации {iteration + 1}")
                break

        computed_readings = A @ x

        output = {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "doserates": self._calculate_doserates(x),
            "effective_readings": {
                name: float(val) for name, val in zip(selected, computed_readings)
            },
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": iteration,
            "method": "doroshenko_matrix",
        }

        return output

    # def unfold_cvxpy(
    #     self,
    #     readings: Dict[str, float],
    #     initial_spectrum: Optional[np.ndarray] = None,
    #     regularization: float = 1e-4,
    #     norm: int = 2,
    #     solver: str = "default",
    #     calculate_errors: bool = False,
    # ) -> Dict:
    #     """
    #     Восстановление спектра с помощью алгоритма cvxpy.

    #     Параметры:
    #     readings : Dict[str, float]
    #         Показания детекторов
    #     initial_spectrum : Optional[np.ndarray], optional
    #         Начальное приближение спектра. Если None, используется равномерный спектр.
    #     regularization : float, optional
    #         Параметр регуляризации. По умолчанию 0.0
    #     calculate_errors : bool, optional
    #         Флаг расчета ошибок восстановления. По умолчанию False

    #     Возвращает:
    #     Dict
    #         Словарь с результатами восстановления спектра.
    #     """
    #     # Валидация и подготовка данных
    #     readings = self._validate_readings(readings)
    #     selected = [name for name in self.detector_names if name in readings]

    #     # Векторизованная подготовка данных
    #     b = np.array([readings[name] for name in selected], dtype=float)
    #     A = np.array([self.sensitivities[name] for name in selected], dtype=float)
    #     n = A.shape[1]
    #     alpha = regularization
    #     x = cp.Variable(n, nonneg=True)
    #     objective = cp.Minimize(cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm))

    #     # Формулируем и решаем задачу
    #     problem = cp.Problem(objective)
    #     if solver == "ECOS":
    #         problem.solve(solver=cp.ECOS)
    #     else:
    #         problem.solve()

    #     # Результаты
    #     print("Статус:", problem.status)
    #     print("Целевая функция:", problem.value)

    #     # Формирование результатов
    #     computed_readings = A @ x.value
    #     print(f"\nНорма невязки: {np.linalg.norm(computed_readings - b):.6f}")

    #     output = {
    #         "energy": self.E_MeV.copy(),
    #         "spectrum": x.value,
    #         "spectrum_absolute": x.value,
    #         "doserates": self._calculate_doserates(x.value),
    #         "effective_readings": dict(zip(selected, computed_readings)),
    #         "residual": b - computed_readings,
    #         "residual norm": np.linalg.norm(b - computed_readings),
    #         "regularization": regularization,
    #         "norm": norm,
    #         "solver": solver,
    #         "method": "cvxpy",
    #     }

    #     if calculate_errors:
    #         print('Calculating uncertainty with Monte-Carlo...')
    #         x_montecarlo = []
    #         # цикл для восстановления спектров с зашумлёнными данными
    #         n_montecarlo = 1000
    #         for i in range(n_montecarlo):
    #             readings_noisy = self._add_noise(readings)
    #             selected = [name for name in self.detector_names if name in readings_noisy]
    #             b = np.array([readings_noisy[name] for name in selected], dtype=float)
    #             A = np.array([self.sensitivities[name] for name in selected], dtype=float)
    #             x = cp.Variable(n, nonneg=True)
    #             objective = cp.Minimize(cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm))
    #             # Формулируем и решаем задачу
    #             problem = cp.Problem(objective)
    #             if solver == "ECOS":
    #                 problem.solve(solver=cp.ECOS)
    #             else:
    #                 problem.solve()
    #             x_montecarlo.append(x.value)

    #         output["spectrum_uncert_mean"] = np.mean(x_montecarlo, axis=0)
    #         output["spectrum_uncert_min"] = np.min(x_montecarlo, axis=0)
    #         output["spectrum_uncert_max"] = np.max(x_montecarlo, axis=0)
    #         output["spectrum_uncert_std"] = np.std(x_montecarlo, axis=0)
    #         print('...uncertainty calculated.')
    #     return output

    def unfold_cvxpy(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "default",
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма cvxpy.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # def _prepare_data(rdgs: Dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        #     """Подготовка матрицы A и вектора b из показаний."""
        #     selected = [name for name in self.detector_names if name in rdgs]
        #     b = np.array([rdgs[name] for name in selected], dtype=float)
        #     A = np.array([self.sensitivities[name] for name in selected], dtype=float)
        #     return A, b, selected

        def _solve_problem(
            A: np.ndarray, b: np.ndarray, use_solver: str = None
        ) -> np.ndarray:
            """Решение задачи оптимизации."""
            x = cp.Variable(A.shape[1], nonneg=True)
            objective = cp.Minimize(cp.norm(A @ x - b, 2) + alpha * cp.norm(x, norm))
            problem = cp.Problem(objective)

            if use_solver == "ECOS":
                problem.solve(solver=cp.ECOS)
            else:
                problem.solve()

            print(f"Статус: {problem.status}")
            print(f"Целевая функция: {problem.value}")
            return x.value

        # Валидация и основное решение
        readings = self._validate_readings(readings)
        A, b, selected = self._build_system(readings)
        alpha = regularization
        n = A.shape[1]

        # Основное решение
        x_value = _solve_problem(A, b, solver)
        computed_readings = A @ x_value
        residual = b - computed_readings
        residual_norm = np.linalg.norm(residual)
        print(f"Норма невязки: {residual_norm:.6f}")

        # Формирование основных результатов
        output = self._standardize_output(
            spectrum=x_value,
            A=A,
            b=b,
            selected=selected,
            method="cvxpy",
            norm = norm,
            solver = solver,
        )

        # Расчет погрешностей методом Монте-Карло (при необходимости)
        if calculate_errors:
            print("Calculating uncertainty with Monte-Carlo...")

            n_montecarlo = 1000
            x_montecarlo = np.empty((n_montecarlo, n))

            # Векторизованный цикл с использованием numpy для ускорения
            for i in range(n_montecarlo):
                readings_noisy = self._add_noise(readings)
                A_noisy, b_noisy, _ = self._build_system(readings_noisy)
                x_montecarlo[i] = _solve_problem(A_noisy, b_noisy, solver)

            output.update(
                {
                    "spectrum_uncert_mean": np.mean(x_montecarlo, axis=0),
                    "spectrum_uncert_min": np.min(x_montecarlo, axis=0),
                    "spectrum_uncert_max": np.max(x_montecarlo, axis=0),
                    "spectrum_uncert_std": np.std(x_montecarlo, axis=0),
                }
            )
            print("...uncertainty calculated.")

        return output

    def unfold_cvxopt(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        norm: int = 2,
        solver: str = "ECOS",
        abstol: float = 1e-10,
        reltol: float = 1e-10,
        feastol: float = 1e-10,
        max_iterations: int = 1000,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма CVXOPT.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        m, n = A.shape

        # Матрица регуляризации: P = A^T A + alpha * I
        P = A.T @ A + regularization * np.eye(n)
        q = -A.T @ b

        # Преобразуем в cvxopt матрицы
        P_cvx = matrix(P.astype(float))
        q_cvx = matrix(q.astype(float))

        # Ограничение неотрицательности
        G = matrix(-np.eye(n).astype(float))
        h = matrix(np.zeros(n).astype(float))

        # Настройки ECOS
        solvers.options.clear()
        solvers.options["abstol"] = abstol
        solvers.options["reltol"] = reltol
        solvers.options["feastol"] = feastol
        solvers.options["show_progress"] = True
        solvers.options["maxit"] = max_iterations

        try:
            if solver == "ECOS":
                solution = solvers.qp(P_cvx, q_cvx, G, h, solver=solver.lower())
        except:
            solvers.options["solver"] = "ecos"
            solution = solvers.qp(P_cvx, q_cvx, G, h)

        if solution["status"] == "optimal":
            x = np.array(solution["x"]).flatten()
        else:
            raise ValueError(f"Решение не найдено. Статус: {solution['status']}")

        # Формирование результатов
        computed_readings = A @ x
        print(f"\nНорма невязки: {np.linalg.norm(computed_readings - b):.6f}")

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "doserates": self._calculate_doserates(x),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "regularization": regularization,
            "max_iterations": max_iterations,
            "norm": norm,
            "solver": solver,
            "method": "cvxopt",
        }

    def unfold_landweber(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма Landweber.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-6
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Создаем пространства ODL
        measurement_space = odl.uniform_discr(0, len(b) - 1, len(b))
        spectrum_space = odl.uniform_discr(
            np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
        )

        # Создаем оператор (матрицу чувствительности)
        operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

        # Преобразуем измерения в ODL вектор
        measurement_vector = measurement_space.element(b)

        # Начальное приближение (равномерный спектр)
        initial_spectrum = spectrum_space.element(0)

        # Применяем метод Ландвебера
        unfolded_spectrum = initial_spectrum.copy()

        # Вычисляем шаг для устойчивости
        step_size = 1.0 / (odl.power_method_opnorm(operator) ** 2)

        for i in range(max_iterations):
            residual = operator(unfolded_spectrum) - measurement_vector
            update = operator.adjoint(residual)
            unfolded_spectrum = unfolded_spectrum - step_size * update

            # Неотрицательность - альтернативный способ
            reconstructed_array = unfolded_spectrum.asarray()
            reconstructed_array[reconstructed_array < 0] = 0
            unfolded_spectrum = spectrum_space.element(reconstructed_array)

        unfolded_spectrum = unfolded_spectrum.data

        # Формирование результатов
        computed_readings = A @ unfolded_spectrum

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": unfolded_spectrum,
            "doserates": self._calculate_doserates(unfolded_spectrum),
            "spectrum_absolute": unfolded_spectrum,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": max_iterations,
            "method": "Landweber",
        }

    def unfold_cuqipy(
        self,
        readings: Dict[str, float],
        readings_error: float = 0.05,
        spectrum_error: float = 0.1,
        method="Gaussian",
        initial_spectrum: Optional[np.ndarray] = None,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма Cuqipy.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        readings_error
        spectrum_error
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        m, n = A.shape

        # Set up Bayesian model for inverse problem
        Amodel = LinearModel(A)  # y = Ax. Model for inverse problem

        if method == "Gaussian":
            x = Gaussian(np.zeros(n), spectrum_error)  # x ~ N(0,0.1)

        elif method == "LMRF":
            x = LMRF(
                0, spectrum_error, geometry=n, bc_type="zero"
            )  # x ~ LMRF(0, 0.01), Zero BC

        elif method == "CMRF":
            x = CMRF(
                0, spectrum_error, geometry=n, bc_type="zero"
            )  # x ~ CMRF(0,0.01), Zero BC

        y = Gaussian(Amodel @ x, readings_error)  # y ~ N(Ax,0.05) cov
        IP = BayesianProblem(y, x).set_data(y=b)  # Bayesian problem given observed data
        samples = IP.UQ()

        Extracted_solution = samples.samples.mean(axis=1)
        Extracted_solution_err = samples.samples.std(axis=1)

        # Формирование результатов
        computed_readings = A @ Extracted_solution
        print(f"\nНорма невязки: {np.linalg.norm(computed_readings - b):.6f}")

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": Extracted_solution,
            "doserates": self._calculate_doserates(Extracted_solution),
            "spectrum_error": Extracted_solution_err,
            "spectrum_absolute": Extracted_solution,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "method": f"CuQipy {method}",
        }

    def unfold_gurobi(
        self,
        readings: Dict[str, float],
        tolerance: float = 1e-6,
        method: str = "MinNormNonnegative",
        initial_spectrum: Optional[np.ndarray] = None,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма Gurobi.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        shape1 = A.shape[1]
        shape0 = A.shape[0]

        try:
            model = gp.Model(method)

            # Неотрицательные переменные
            x = model.addVars(shape1, lb=0.0, name="x")

            # Ограничения: Ax = b (с допуском)
            for i in range(shape0):
                expr = gp.LinExpr()
                for j in range(shape1):
                    expr += A[i, j] * x[j]
                model.addConstr(expr >= b[i] - tolerance, f"constr_lower_{i}")
                model.addConstr(expr <= b[i] + tolerance, f"constr_upper_{i}")

            # Целевая функция: минимизация нормы x
            obj = gp.QuadExpr()
            for j in range(shape1):
                obj += x[j] * x[j]

            model.setObjective(obj, GRB.MINIMIZE)
            model.optimize()

            if model.status == GRB.OPTIMAL:
                x_sol = np.array([x[i].X for i in range(shape1)])

                # Формирование результатов
                computed_readings = A @ x_sol

                print(f"\nНорма невязки: {np.linalg.norm(computed_readings - b):.6f}")
                print("Решение с минимальной нормой найдено!")
                print(f"Ненулевых компонент: {np.sum(x_sol > tolerance)}")
                print(f"Норма решения: {np.linalg.norm(x_sol):.6f}")
            else:
                print(f"Решение не найдено. Статус: {model.status}")
                x_sol = None
                return None

        except Exception as e:
            print(f"Ошибка: {e}")
            return None

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x_sol,
            "spectrum_absolute": x_sol,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "doserates": self._calculate_doserates(x_sol),
            "tolarance": tolerance,
            "method": f"gurobi {method}",
        }

    # нужно отмасштабировать
    # def unfold_conjugate_gradient(
    #     self,
    #     readings: Dict[str, float],
    #     initial_spectrum: Optional[np.ndarray] = None,
    #     tolerance: float = 1e-6,
    #     max_iterations: int = 1000,
    #     regularization: float = 0.0,
    #     calculate_errors: bool = False,
    # ) -> Dict:
    #     """
    #     Восстановление спектра с помощью алгоритма conjugate_gradient.

    #     Параметры:
    #     readings : Dict[str, float]
    #         Показания детекторов
    #     initial_spectrum : Optional[np.ndarray], optional
    #         Начальное приближение спектра. Если None, используется равномерный спектр.
    #     tolerance : float, optional
    #         Точность сходимости. По умолчанию 1e-6
    #     max_iterations : int, optional
    #         Максимальное число итераций. По умолчанию 1000
    #     regularization : float, optional
    #         Параметр регуляризации. По умолчанию 0.0
    #     calculate_errors : bool, optional
    #         Флаг расчета ошибок восстановления. По умолчанию False

    #     Возвращает:
    #     Dict
    #         Словарь с результатами восстановления спектра.
    #     """

    #     # логирование
    #     callback = odl.solvers.CallbackPrintIteration()

    #     # Валидация и подготовка данных
    #     readings = self._validate_readings(readings)
    #     selected = [name for name in self.detector_names if name in readings]

    #     # Векторизованная подготовка данных
    #     b = np.array([readings[name] for name in selected], dtype=float)
    #     A = np.array([self.sensitivities[name] for name in selected], dtype=float)

    #     # Создаем пространства ODL
    #     measurement_space = odl.uniform_discr(0, len(b) - 1, len(b))
    #     spectrum_space = odl.uniform_discr(
    #         np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
    #     )

    #     # Создаем оператор (матрицу чувствительности)
    #     operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

    #     y = measurement_space.element(b)
    #     if initial_spectrum is None:
    #         # Начальное приближение
    #         x = spectrum_space.element(0.5)
    #     else:
    #         x = spectrum_space.element(initial_spectrum)

    #     odl.solvers.conjugate_gradient_normal(
    #         operator, x, y, niter=max_iterations, callback=callback
    #     )

    #     # Применяем неотрицательность
    #     x_arr = x.asarray()
    #     x_arr[x_arr < 0] = 0
    #     unfolded_spectrum = x_arr
    #     # Формирование результатов
    #     computed_readings = A @ unfolded_spectrum

    #     return {
    #         "energy": self.E_MeV.copy(),
    #         "spectrum": unfolded_spectrum,
    #         "spectrum_absolute": unfolded_spectrum,
    #         "effective_readings": dict(zip(selected, computed_readings)),
    #         "iterations": max_iterations,
    #         "method": "Conjugate gradients",
    #     }

    def unfold_gauss_newton(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        regularization: float = 0.0,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма gauss_newton.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-6
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000
        regularization : float, optional
            Параметр регуляризации. По умолчанию 0.0
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # логирование
        callback = odl.solvers.CallbackPrintIteration()

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Создаем пространства ODL
        measurement_space = odl.uniform_discr(0, len(b) - 1, len(b))
        spectrum_space = odl.uniform_discr(
            np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
        )

        # Создаем оператор (матрицу чувствительности)
        operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

        y = measurement_space.element(b)
        if initial_spectrum is None:
            # Начальное приближение
            x = spectrum_space.element(0.5)
        else:
            x = spectrum_space.element(initial_spectrum)

        odl.solvers.iterative.iterative.gauss_newton(
            operator, x, y, niter=max_iterations, callback=callback
        )

        # Применяем неотрицательность
        x_arr = x.asarray()
        x_arr[x_arr < 0] = 0
        unfolded_spectrum = x_arr
        # Формирование результатов
        computed_readings = A @ unfolded_spectrum

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": unfolded_spectrum,
            "spectrum_absolute": unfolded_spectrum,
            "doserates": self._calculate_doserates(unfolded_spectrum),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": max_iterations,
            "method": "Gauss-Newton",
        }

    def unfold_kaczmarz(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        omega: float = 1,
        calculate_errors: bool = False,
    ) -> Dict:
        """
         Восстановление спектра с помощью алгоритма Качмажа.

         Параметры:
         readings : Dict[str, float]
             Показания детекторов
         initial_spectrum : Optional[np.ndarray], optional
             Начальное приближение спектра. Если None, используется равномерный спектр.
         tolerance : float, optional
             Точность сходимости. По умолчанию 1e-6
         max_iterations : int, optional
             Максимальное число итераций. По умолчанию 1000

        omega - positive float or sequence of positive floats, optional
         Relaxation parameter in the iteration.
         If a single float is given the same step is used for all operators,
         otherwise separate steps are used.

         calculate_errors : bool, optional
             Флаг расчета ошибок восстановления. По умолчанию False

         Возвращает:
         Dict
             Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        Amat = np.array([self.sensitivities[name] for name in selected], dtype=float)
        m, n = Amat.shape

        if initial_spectrum is None:
            x = np.zeros(n)
        else:
            x = initial_spectrum.copy()

        history = []

        for k in range(max_iterations):
            # Циклический проход по строкам
            i = k % m
            a_i = Amat[i, :]
            denominator = np.dot(a_i, a_i)

            if denominator > 0:
                # Обновление решения
                update = (b[i] - np.dot(a_i, x)) / denominator
                x = x + omega * update * a_i

            # Сохраняем ошибку
            if k % 10 == 0:
                error = np.linalg.norm(Amat @ x - b) / np.linalg.norm(b)
                history.append(error)

        # Формирование результатов
        computed_readings = Amat @ x

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "doserates": self._calculate_doserates(x),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": max_iterations,
            "omega": omega,
            "method": "Качмаж",
        }

    def unfold_kaczmarz2(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 4000,
        rule: str = "maxdistance",
        calculate_errors: bool = False,
    ) -> Dict:
        """
         Восстановление спектра с помощью алгоритма Качмажа.
         Автор: Jacob Moorman
         pip install -U kaczmarz-algorithms
         https://github.com/jdmoorman/kaczmarz-algorithms

         Параметры:
         readings : Dict[str, float]
             Показания детекторов
         initial_spectrum : Optional[np.ndarray], optional
             Начальное приближение спектра. Если None, используется равномерный спектр.
         tolerance : float, optional
             Точность сходимости. По умолчанию 1e-6
         max_iterations : int, optional
             Максимальное число итераций. По умолчанию 1000

         rule : str, optional  = maxdistance, cyclic

        omega - positive float or sequence of positive floats, optional
         Relaxation parameter in the iteration.
         If a single float is given the same step is used for all operators,
         otherwise separate steps are used.

         calculate_errors : bool, optional
             Флаг расчета ошибок восстановления. По умолчанию False

         Возвращает:
         Dict
             Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        if initial_spectrum is None:
            x = np.zeros(A.shape[1])
        else:
            x = initial_spectrum.copy()

        if rule == "maxdistance":
            x = kaczmarz.MaxDistance.solve(A, b, maxiter=max_iterations, tol=tolerance)
        elif rule == "cyclic":
            x = kaczmarz.Cyclic.solve(A, b, maxiter=max_iterations, tol=tolerance)

        # Формирование результатов
        computed_readings = A @ x

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "doserates": self._calculate_doserates(x),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": max_iterations,
            "method": f"Качмаж, {rule}",
        }

    def unfold_mlem_odl(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-6,
        max_iterations: int = 1000,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма Maximum
        Likelihood Expectation Maximation algorithm,
        poisson_log_likelihood  - Poisson log-likelihood.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        tolerance : float, optional
            Точность сходимости. По умолчанию 1e-6
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000

        lambda_relax - positive float or sequence of positive floats, optional
        Relaxation parameter in the iteration.
        If a single float is given the same step is used for all operators,
        otherwise separate steps are used.

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # логирование
        callback = odl.solvers.CallbackPrintIteration()

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Создаем пространства ODL
        measurement_space = odl.uniform_discr(0, len(b) - 1, len(b))
        spectrum_space = odl.uniform_discr(
            np.min(self.E_MeV), np.max(self.E_MeV), self.E_MeV.shape[0]
        )

        # Создаем оператор (матрицу чувствительности)
        operator = odl.MatrixOperator(A, domain=spectrum_space, range=measurement_space)

        y = measurement_space.element(b)
        if initial_spectrum is None:
            # Начальное приближение
            x = spectrum_space.element(
                0.5
            )  # 0.5 - хорошо, иначе при 1 или 0 получаем нули.
        else:
            x = spectrum_space.element(initial_spectrum)

        odl.solvers.mlem(operator, x, y, niter=max_iterations, callback=callback)

        unfolded_spectrum = x.asarray()
        # Формирование результатов
        computed_readings = A @ unfolded_spectrum

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": unfolded_spectrum,
            "spectrum_absolute": unfolded_spectrum,
            "doserates": self._calculate_doserates(unfolded_spectrum),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": max_iterations,
            "method": "MLEM (ODL)",
        }

    def unfold_evolutionary(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1,
        penalty_weight: int = 1000,
        population_size: int = 3000,
        generations: int = 3000,
        mu: float = 0,
        mate_param: float = 0.3,
        tournsize: int = 3,
        mutation_strength: float = 0.3,
        individ_probability: float = 0.1,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью эволюционный алгоритм с регуляризацией

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        penalty_weight - штраф за отрицательные значения
        regularization - параметр регуляризации по Тихонову

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        ## **Параметры мутации:**

        ### `mu` (mean) - среднее значение
        - **Назначение**: Определяет центр распределения для мутации
        - **Значение 0**: Мутация в среднем не изменяет значение (равновероятно увеличивает и уменьшает)
        - **Пример**: `mu=0` - симметричная мутация

        ### mutation_strength = `sigma` (standard deviation) - стандартное отклонение
        - **Назначение**: Определяет "силу" мутации (разброс изменений)
        - **Значение 0.5**: Средняя интенсивность мутации
        - **Чем больше sigma**: Тем сильнее могут изменяться значения
        - **Пример**:
        - `sigma=0.1` - мелкие изменения
        - `sigma=1.0` - крупные изменения

        ### individ_probability = `indpb` (individual probability) - вероятность мутации особи
        - **Назначение**: Вероятность того, что каждый ген особи будет мутировать
        - **Значение 0.1**: Каждый ген имеет 10% шанс быть мутированным
        - **Диапазон**: от 0.0 (нет мутации) до 1.0 (все гены мутируют)
        - **Пример**: Для особи из 46 генов при `indpb=0.1` в среднем мутируют 4-5 генов

        ## **Параметры скрещивания (mate):**

        ### mate_param = `alpha` - параметр смешивания
        - **Назначение**: Определяет, как комбинируются гены родителей
        - **Формула**: `child = parent1 * alpha + parent2 * (1 - alpha)`
        - **Значение 0.3**: Потомок получает 30% от первого родителя и 70% от второго
        - **Диапазон**: обычно от 0.0 до 1.0
        - **Особенности**:
        - `alpha=0.0` - потомок идентичен второму родителю
        - `alpha=0.5` - равномерное смешивание
        - `alpha=1.0` - потомок идентичен первому родителю

        ## **Параметры селекции:**

        ### `tournsize` - размер турнира
        - **Назначение**: Количество особей, участвующих в каждом турнире
        - **Значение 3**: Из 3 случайных особей выбирается лучшая
        - **Чем больше tournsize**: Тем более жесткий отбор
        - **Влияние**:
        - Маленький турнир (`tournsize=2`) - слабый отбор, больше разнообразия
        - Большой турнир (`tournsize=7`) - сильный отбор, быстрая сходимость

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        def evaluate_solution(individual, A, b):
            """
            Функция оценки решения - минимизирует невязку ||Ax - b||^2
            с учетом неотрицательности решения
            """
            # Преобразуем в неотрицательные значения
            x = np.maximum(0, np.array(individual))
            residual = A @ x - b
            error = np.sum(residual**2)  # L2 норма невязки
            return (error,)

        def evaluate_solution_regularized(individual, A, b, alpha=regularization):
            """
            Функция оценки с регуляризацией Тихонова
            Минимизирует ||Ax - b||^2 + alpha * ||x||^2
            """
            x = np.maximum(0, np.array(individual))
            residual = A @ x - b
            error = np.sum(residual**2) + alpha * np.sum(x**2)
            return (error,)

        def evaluate_solution_penalty(individual, A, b, penalty_weight=penalty_weight):
            """
            Функция оценки с штрафом за отрицательные значения
            """
            x = np.array(individual)

            # Штраф за отрицательные значения
            negative_penalty = penalty_weight * np.sum(np.minimum(0, x) ** 2)

            residual = A @ x - b
            error = np.sum(residual**2) + negative_penalty
            return (error,)

        def solve_nonnegative_system(
            A,
            b,
            mu=mu,
            mate_param=mate_param,
            tournsize=tournsize,
            mutation_strength=mutation_strength,
            individ_probability=individ_probability,
            population_size=population_size,
            generations=generations,
            use_penalty=True,
        ):
            """
            Решает недоопределенную систему уравнений с ограничением неотрицательности
            """
            # Создаем типы для генетического алгоритма
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()

            # Регистрируем функции для создания популяции
            # Начальные значения в положительном диапазоне
            toolbox.register("attr_float", random.uniform, 0.0, 5.0)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.attr_float,
                n=A.shape[1],
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Выбираем функцию оценки
            if use_penalty:
                toolbox.register("evaluate", evaluate_solution_penalty, A=A, b=b)
            else:
                toolbox.register("evaluate", evaluate_solution_regularized, A=A, b=b)

            # Специальные операторы для неотрицательных решений
            toolbox.register("mate", tools.cxBlend, alpha=mate_param)
            toolbox.register(
                "mutate",
                nonnegative_mutation,
                mu=mu,
                sigma=mutation_strength,
                indpb=individ_probability,
            )
            toolbox.register("select", tools.selTournament, tournsize=tournsize)

            # Создаем начальную популяцию
            population = toolbox.population(n=population_size)

            # Собираем статистику
            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            # Дополнительная статистика по неотрицательности
            stats_neg = tools.Statistics(lambda ind: count_negative_values(ind))
            stats_neg.register("neg_count", np.mean)

            mstats = tools.MultiStatistics(fitness=stats, negativity=stats_neg)

            # Запускаем генетический алгоритм
            population, logbook = algorithms.eaSimple(
                population,
                toolbox,
                cxpb=mate_param,  # 0.7
                mutpb=mutation_strength,  # 0.3
                ngen=generations,
                stats=mstats,
                verbose=True,
            )

            # Выбираем лучшее решение и применяем неотрицательность
            best_individual = tools.selBest(population, k=1)[0]
            best_solution = np.maximum(
                0, np.array(best_individual)
            )  # Гарантируем неотрицательность

            return best_solution, logbook

        def nonnegative_mutation(individual, mu, sigma, indpb):
            """
            Мутация с гарантией неотрицательности
            """
            for i in range(len(individual)):
                if random.random() < indpb:
                    # Мутируем, но не позволяем уйти глубоко в отрицательную область
                    new_value = individual[i] + random.gauss(mu, sigma)
                    individual[i] = max(0, new_value)  # Обрезаем отрицательные значения
            return (individual,)

        def count_negative_values(individual):
            """
            Подсчитывает количество отрицательных значений в решении
            """
            return sum(1 for x in individual if x < 0)

        def repair_negative_values(individual):
            """
            Исправляет отрицательные значения в решении
            """
            return [max(0, x) for x in individual]

        def solve_with_projections(
            A,
            b,
            mu=mu,
            mate_param=mate_param,
            tournsize=tournsize,
            mutation_strength=mutation_strength,
            individ_probability=individ_probability,
            population_size=population_size,
            generations=generations,
        ):
            """
            Альтернативный подход с проекцией на неотрицательную область
            """
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, 0.0, 3.0)
            toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                toolbox.attr_float,
                n=A.shape[1],
            )
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", evaluate_solution_regularized, A=A, b=b)
            toolbox.register("mate", tools.cxBlend, alpha=mate_param)
            toolbox.register(
                "mutate",
                tools.mutGaussian,
                mu=mu,
                sigma=mutation_strength,
                indpb=individ_probability,
            )
            toolbox.register("select", tools.selTournament, tournsize=tournsize)

            # Добавляем оператор восстановления после скрещивания и мутации
            toolbox.decorate("mate", tools.DeltaPenalty(repair_negative_values, 1.0))
            toolbox.decorate("mutate", tools.DeltaPenalty(repair_negative_values, 1.0))

            population = toolbox.population(n=population_size)

            stats = tools.Statistics(lambda ind: ind.fitness.values[0])
            stats.register("avg", np.mean)
            stats.register("min", np.min)

            population, logbook = algorithms.eaSimple(
                population,
                toolbox,
                cxpb=mate_param,  # 0.7
                mutpb=mutation_strength,  # 0.2
                ngen=generations,
                stats=stats,
                verbose=True,
            )

            best_individual = tools.selBest(population, k=1)[0]
            best_solution = np.array(best_individual)

            return best_solution, logbook

        x_final, logbook = solve_nonnegative_system(
            A, b, use_penalty=True, generations=generations
        )

        # Формирование результатов
        computed_readings = A @ x_final

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x_final,
            "spectrum_absolute": x_final,
            "doserates": self._calculate_doserates(x_final),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "method": "Evolutionary (DEAP)",
            "population_size": f"{population_size}",
            "generations": f"{generations}",
            "regularization": regularization,
            "mu": mu,
            "mate_param": mate_param,
            "tournsize": tournsize,
            "mutation_strength": mutation_strength,
            "individ_probability": individ_probability,
            "logbook": logbook,
        }

    def unfold_bayes(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        The unfolding method implemented in PyUnfold accomplishes this deconvolution by harnessing
        the power of Bayes’ Theorem in an iterative procedure, providing results based on physical
        expectations of the desired quantity
        G D’Agostini, “A Multidimensional unfolding method based on Bayes’ theorem”,
        Nucl. Instrum. Meth. A 362 (1995) 487.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Detection efficiencies
        efficiencies = [1] * A.shape[1]

        # сделать вариант кода с учётом ошибок
        response_err = np.zeros_like(A)  # ошибки нулевые в ФЧ
        efficiencies_err = [0.05] * A.shape[
            1
        ]  # ошибки 0.05 в эффективности регистрации
        data_err = [0.05] * A.shape[0]  # ошибки [0.05] в данных

        # Perform iterative unfolding
        result = iterative_unfold(
            data=b,
            data_err=data_err,
            response=A,
            response_err=response_err,
            efficiencies=efficiencies,
            efficiencies_err=efficiencies_err,
            max_iter=max_iterations,
            callbacks=[Logger()],
            prior=initial_spectrum,  # априорный спектр с которого начинаем воостановление.
            ts_stopping=tolerance,
        )

        computed_readings = A @ result["unfolded"]

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": result["unfolded"],
            "spectrum statistical uncertainies": result["stat_err"],
            "systematic uncertainties": result["sys_err"],
            "spectrum_absolute": result["unfolded"],
            "doserates": self._calculate_doserates(result["unfolded"]),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": result["num_iterations"],
            "method": "Bayes theorem (D’Agostini)",
        }

    def unfold_bayes_spline_regularization(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        max_iterations: int = 4000,
        tolerance: float = 1e-3,
        spline_degree: int = 3,
        spline_smooth: float = 1e-2,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        The unfolding method implemented in PyUnfold accomplishes this deconvolution by harnessing
        the power of Bayes’ Theorem in an iterative procedure, providing results based on physical
        expectations of the desired quantity
        G D’Agostini, “A Multidimensional unfolding method based on Bayes’ theorem”,
        Nucl. Instrum. Meth. A 362 (1995) 487.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        max_iterations : int, optional
            Максимальное число итераций. По умолчанию 1000

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        spline_reg = SplineRegularizer(degree=spline_degree, smooth=spline_smooth)

        # Detection efficiencies
        efficiencies = [1] * A.shape[1]

        # сделать вариант кода с учётом ошибок
        response_err = np.zeros_like(A)  # ошибки нулевые в ФЧ
        efficiencies_err = [0.05] * A.shape[
            1
        ]  # ошибки 0.05 в эффективности регистрации
        data_err = [0.05] * A.shape[0]  # ошибки [0.05] в данных

        # Perform iterative unfolding
        result = iterative_unfold(
            data=b,
            data_err=data_err,
            response=A,
            response_err=response_err,
            efficiencies=efficiencies,
            efficiencies_err=efficiencies_err,
            max_iter=max_iterations,
            callbacks=[Logger(), spline_reg],
            prior=initial_spectrum,  # априорный спектр с которого начинаем воостановление.
            ts_stopping=tolerance,
        )

        computed_readings = A @ result["unfolded"]

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": result["unfolded"],
            "spectrum statistical uncertainies": result["stat_err"],
            "systematic uncertainties": result["sys_err"],
            "spectrum_absolute": result["unfolded"],
            "doserates": self._calculate_doserates(result["unfolded"]),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "iterations": result["num_iterations"],
            "spline_degree": spline_degree,
            "spline_smooth": spline_smooth,
            "method": "Bayes theorem (D’Agostini) with spline regularization",
        }

    def unfold_statreg(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        unfoldermethod: str = "EmpiricalBayes",
        regularization: Optional[np.ndarray] = None,
        basis_name: str = "CubicSplines",
        boundary: str = None,
        derivative_degree: int = 2,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        Python implementation of Turchin's method of statistical regularization
        - для корректной работы необходимо изменить код импортируемого модуля statreg
        - ноутбук с регуляризацией Турчина в исполнении пакета statreg, Внимание:
        .venv/lib/python3.13/site-packages/statreg/model/gauss_error.py
        - строка 386 return np.transpose(np.compress(null_mask, vh, axis=0)),
        поменял sc. на np. - чтобы код работал под новой версией scipy
        1. Zelenyi M, Poliakova M, Nozik A, Khudyakov A. Application of
        Turchin’s method of statistical regularization. Verkheev A, Aparin A,
        Bobrikov I, Chudoba V, Friesen A, editors. EPJ Web Conf. 2018;177:07005.
        https://doi.org/10.1051/epjconf/201817707005

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        boundary :
            Boundary condition. Allowed boundary conditions are:

            - None         : no boundary conditions (default)
            - "dirichlet"  : function must be 0 at border

        derivative_degree
            степень производной 1,2,3

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # сделать код решения с учётом ошибок
        b_err = b * 0.05  # np.sqrt(b)   # ошибки [0.05] в данных или корень из них
        Emin = np.min(self.E_MeV)

        if basis_name == "CubicSplines":
            basis = CubicSplines(
                np.log10(self.E_MeV / Emin), boundary=boundary
            )  # Создание базиса кубических сплайнов np.array(np.log10(ebins/Emin))

        # для boundary = dirichlet пока не работает

        omega = basis.omega(derivative_degree)  # регуляризация по производной

        # Создание модели и решение
        if unfoldermethod == "EmpiricalBayes":
            model = GaussErrorMatrixUnfolder(omega, method=unfoldermethod)
        if unfoldermethod == "User":
            if regularization is None:  # если явно не указано, то берем 1e-4
                regularization = 1e-4
            model = GaussErrorMatrixUnfolder(
                omega, method=unfoldermethod, alphas=regularization
            )

        result = model.solve(A, b, b_err)

        computed_readings = A @ result.phi

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": result.phi,
            "covariance": result["covariance"],
            "uncertainties": np.sqrt(np.diag(result["covariance"])),
            "alphas": result["alphas"],
            "spectrum_absolute": result.phi,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "doserates": self._calculate_doserates(result.phi),
            "unfolder": unfoldermethod,
            "basis_name": basis_name,
            "boundary": boundary,
            "derivative_degree": derivative_degree,
            "method": "Turchin's method of statistical regularization",
        }

    def unfold_scipy_direct_method(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        tolerance: float = 1e-8,
        max_iterations: int = 4000,
        method: str = "cg",
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        tolerance
        max_iterations
        method = 'cg',cgs, bicgstab,gmres, lgmres,minres, gcrotmk,qmr, tfqmr, lsmr, lsqr

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        A, b, selected = self._build_system(readings)
        AT_A = A.T @ A
        AT_b = A.T @ b

        solvers = {
            "cg": lambda: cg(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "cgs": lambda: cgs(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "bicgstab": lambda: bicgstab(
                AT_A, AT_b, rtol=tolerance, maxiter=max_iterations
            ),
            "gmres": lambda: gmres(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "lgmres": lambda: lgmres(
                AT_A, AT_b, rtol=tolerance, maxiter=max_iterations
            ),
            "minres": lambda: minres(
                AT_A, AT_b, rtol=tolerance, maxiter=max_iterations
            ),
            "qmr": lambda: qmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "gcrotmk": lambda: gcrotmk(
                AT_A, AT_b, rtol=tolerance, maxiter=max_iterations
            ),
            "tfqmr": lambda: tfqmr(AT_A, AT_b, rtol=tolerance, maxiter=max_iterations),
            "lsqr": lambda: lsqr(A, b, atol=tolerance),
            "lsmr": lambda: lsmr(A, b, atol=tolerance, maxiter=max_iterations),
        }

        if method not in solvers:
            raise ValueError(f"Unknown method: {method}")

        x = solvers[method]()[0]
        x = np.maximum(x, 0)  # Enforce non-negativity

        computed_readings = A @ x

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "effective_readings": dict(zip(selected, computed_readings)),
            "doserates": self._calculate_doserates(x),
            "residual": b - computed_readings,
            "residual norm": np.linalg.norm(b - computed_readings),
            "unfolder": method,
            "iterations": max_iterations,
            "tolerance": tolerance,
            "method": f"Scipy direct method {method}",
        }

    def unfold_mcmc(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        prior_type: str = "gamma",
        n_samples: int = 5000,
        cpucores=None,
        tune: int = 5000,
        target_accept: float = 0.9,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        PyMC is a probabilistic programming library for Python that allows users
        to build Bayesian models with a simple Python API and fit them using Markov
        chain Monte Carlo (MCMC) methods.
        https://www.pymc.io/welcome.html

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        Решение некорректной системы Ax = b с ограничением x >= 0

        Parameters:
        A : array, shape (m, n) - матрица коэффициентов
        b : array, shape (n,) - вектор правой части
        prior_type : str - тип априорного распределения
            'truncated_normal', 'exponential', 'half_normal', 'gamma'

        Returns:
        trace - объект с результатами сэмплирования


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        m = A.shape[1]

        with pm.Model() as model:
            if prior_type == "truncated_normal":
                # Усеченное нормальное распределение
                x = pm.TruncatedNormal("x", mu=0, sigma=1, lower=0, shape=m)

            elif prior_type == "exponential":
                # Экспоненциальное распределение
                x = pm.Exponential("x", lam=1, shape=m)

            elif prior_type == "half_normal":
                # Полунормальное распределение
                x = pm.HalfNormal("x", sigma=1, shape=m)

            elif prior_type == "gamma":
                # Гамма распределение
                x = pm.Gamma("x", alpha=1, beta=1, shape=m)

            # Предсказание b
            b_pred = pm.math.dot(A, x)

            # Неопределенность наблюдений
            sigma = pm.HalfNormal("sigma", sigma=1)

            # Правдоподобие
            likelihood = pm.Normal("b_obs", mu=b_pred, sigma=sigma, observed=b)

            # Сэмплирование
            # cores The number of chains to run in parallel.
            trace = pm.sample(
                n_samples, tune=tune, target_accept=target_accept, cores=cpucores
            )

            x_samples = np.array(trace.posterior["x"])
            result = np.array(trace.posterior["x"].mean(axis=(0, 1)))

        computed_readings = A @ result
        residual = b - computed_readings
        residual_norm = np.linalg.norm(residual)

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": result,
            "spectrum_absolute": result,
            "doserates": self._calculate_doserates(result),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": residual,
            "residual norm": residual_norm,
            "prior_type": prior_type,
            "n_samples": n_samples,
            "method": "Markov chain Monte Carlo (MCMC)",
        }

    def unfold_ridge(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "ridgecv",
        regularization: float = 1e-4,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма
        https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        Note that the class Ridge allows for the user to specify that the solver be
        automatically chosen by setting solver="auto". When this option is specified,
        Ridge will choose between the "lbfgs", "cholesky", and "sparse_cg" solvers
        - https://scikit-learn.org/stable/modules/linear_model.html

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        if method == "ridge":
            if regularization == None:
                regularization = 1e-4

            reg = linear_model.Ridge(alpha=regularization, positive=True)

        if method == "ridgecv":
            reg = linear_model.RidgeCV(
                alphas=np.logspace(-19, 2, 50)
            )  # ищем оптимальный параметр регулризации

        if method == "lasso":
            reg = linear_model.Lasso(alpha=regularization, positive=True)

        if method == "bayesianridge":
            reg = linear_model.BayesianRidge(
                alpha_init=regularization, lambda_init=regularization
            )

        reg.fit(A, b)

        if method == "ridgecv":
            regularization = reg.alpha_  # сохраняем подобранный коэффициент

        computed_readings = A @ reg.coef_
        residual = b - computed_readings
        residual_norm = np.linalg.norm(residual)

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": reg.coef_,
            "spectrum_absolute": reg.coef_,
            "doserates": self._calculate_doserates(reg.coef_),
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": residual,
            "residual norm": residual_norm,
            "regularization": regularization,
            "regularization method": method,
            "method": f"Sklearn Ridge, {method}",
        }

    def unfold_pyomo(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        solver_name: str = "gurobi",
        regularization: float = 1e-4,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью решателей pyomo
        # Линейное программирование (LP)
        'glpk': 'Линейное/Целочисленное (LP/MILP)',
        'cbc': 'Линейное/Целочисленное (LP/MILP)',
        'gurobi': 'Линейное/Целочисленное/Квадратичное (LP/MILP/QP)',
        'cplex': 'Линейное/Целочисленное/Квадратичное (LP/MILP/QP)',
        'xpress': 'Линейное/Целочисленное/Квадратичное (LP/MILP/QP)',
        'highs': 'Линейное/Целочисленное (LP/MILP)',

        # Нелинейное программирование (NLP)
        'ipopt': 'Нелинейное (NLP)',
        'bonmin': 'Смешанное целочисленное нелинейное (MINLP)',
        'couenne': 'Глобальная нелинейная оптимизация (MINLP)',
        'knitro': 'Нелинейное/Целочисленное (NLP/MINLP)',
        'baron': 'Глобальная нелинейная оптимизация (MINLP)',
        'conopt': 'Нелинейное (NLP)',
        'snopt': 'Нелинейное (NLP)',

        # Квадратичное программирование (QP)
        'osqp': 'Квадратичное (QP)',
        'proxqp': 'Квадратичное (QP)',

        # Смешанное целочисленное нелинейное программирование (MINLP)
        'scip': 'Смешанное целочисленное нелинейное (MINLP)',

        # Специализированные
        'path': 'Дополняющие задачи',
        'mosek': 'Коническая оптимизация',
        'octeract': 'Глобальная нелинейная оптимизация',
        'mindtpy': 'Декомпозиция MINLP',
        'pysp': 'Стохастическое программирование',

        installation: https://datawookie.dev/blog/2024/12/optimisation-with-pyomo/
        https://github.com/ampl/amplpy

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.

        Note that the class Ridge allows for the user to specify that the solver be
        automatically chosen by setting solver="auto". When this option is specified,
        Ridge will choose between the "lbfgs", "cholesky", and "sparse_cg" solvers
        - https://scikit-learn.org/stable/modules/linear_model.html

        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        model = pyo.ConcreteModel()
        model.name = "NonnegativeUnderdeterminedSystem"
        # Множества индексов
        model.I = pyo.Set(initialize=range(A.shape[0]))  # уравнения
        model.J = pyo.Set(initialize=range(A.shape[1]))  # переменные

        # Переменные решения с ограничением неотрицательности
        model.x = pyo.Var(model.J, domain=pyo.NonNegativeReals)

        # Целевая функция: минимизация ||Ax - b||^2 + regularization * ||x||^2
        def objective_rule(model):
            # Часть 1: ||Ax - b||^2
            residual_sum = 0.0
            for i in model.I:
                linear_comb = 0.0
                for j in model.J:
                    linear_comb += A[i, j] * model.x[j]
                residual_sum += (linear_comb - b[i]) ** 2

            # Часть 2: регуляризация ||x||^2
            reg_sum = 0.0
            for j in model.J:
                reg_sum += regularization * model.x[j] ** 2

            return residual_sum + reg_sum

        model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

        # Решаем задачу
        solver = SolverFactory(
            solver_name
        )  # можно использовать 'gurobi' если установлен
        results = solver.solve(model, tee=False)  # tee=True для вывода процесса

        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Извлекаем решение
            x = np.array([pyo.value(model.x[j]) for j in range(A.shape[1])])
            computed_readings = A @ x
            residual = b - computed_readings
        else:
            x = None
            computed_readings = None
            residual = None

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": residual,
            "residual norm": np.linalg.norm(residual),
            "solution norm": np.linalg.norm(x),
            "solver": f"{solver_name}",
            "doserates": self._calculate_doserates(x),
            "method": f"Pyomo, {solver_name}",
        }

    def unfold_lmfit(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "leastsq",
        model_name: str = "elastic",
        regularization: float = 1e-4,
        regularization2: float = 1e-4,
        l1_weight: float = 0.5,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью решателей lmfit

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        method : str
            Метод оптимизации
        model_name : str
            Модель регуляризации ('elastic', 'lasso', 'ridge')
        regularization : float
            Параметр регуляризации для L1
        regularization2 : float
            Параметр регуляризации для L2
        l1_weight : float
            Вес L1 регуляризации в Elastic Net
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления


            'leastsq': Levenberg-Marquardt (default)
            'least_squares': Least-Squares minimization, using Trust Region Reflective method
            'differential_evolution': differential evolution
            'brute': brute force method
            'basinhopping': basinhopping
            'ampgo': Adaptive Memory Programming for Global Optimization
            'nelder': Nelder-Mead
            'lbfgsb': L-BFGS-B
            'powell': Powell
            'cg': Conjugate-Gradient
            'newton': Newton-CG - jakobian needed
            'cobyla': Cobyla
            'bfgs': BFGS
            'tnc': Truncated Newton - jakobian needed
            'trust-ncg': Newton-CG trust-region - jakobian needed
            'trust-exact': nearly exact trust-region - jakobian needed
            'trust-krylov': Newton GLTR trust-region - jakobian needed
            'trust-constr': trust-region for constrained optimization - jakobian needed
            'dogleg': Dog-leg trust-region
            'slsqp': Sequential Linear Squares Programming
            'emcee': Maximum likelihood via Monte-Carlo Markov Chain
            'shgo': Simplicial Homology Global Optimization
            'dual_annealing': Dual Annealing optimization


        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """

        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)
        m = A.shape[1]

        # Создаем параметры с ограничением неотрицательности и начальными значениями
        params = lmfit.Parameters()

        # Устанавливаем начальные значения
        if initial_spectrum is not None:
            if len(initial_spectrum) != m:
                raise ValueError(f"initial_spectrum должен иметь длину {m}")
            init_values = initial_spectrum
        else:
            # Равномерное начальное приближение
            init_values = np.ones(m) * np.mean(b) / np.mean(A.sum(axis=1))

        for i in range(m):
            params.add(
                f"x{i}",
                value=max(init_values[i], 1e-10),  # Избегаем нулевых начальных значений
                min=0.0,
                max=None,
            )

        # Определение целевых функций для разных моделей
        def residual_lasso(params, A, b, regularization):
            """Невязка для L1-регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b
            # Возвращаем вектор невязок для метода leastsq или скаляр для других методов
            if method == "leastsq":
                # Для leastsq возвращаем расширенный вектор невязок
                reg_residual = np.sqrt(regularization) * np.sqrt(m) * x
                return np.concatenate([residual, reg_residual])
            else:
                # Для других методов возвращаем скалярную целевую функцию
                return np.sum(residual**2) + regularization * np.sum(np.abs(x))

        def jacobian_lasso(params, A, b, regularization):
            """Якобиан для L1-регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            # n_detectors = A.shape[0]

            # Градиент основной части
            grad_main = 2 * A.T @ (A @ x - b)

            # Градиент L1 регуляризации (субградиент в точке 0)
            grad_reg = regularization * np.sign(x)
            # Для x=0 используем 0 в субградиенте
            grad_reg[np.abs(x) < 1e-10] = 0

            return grad_main + grad_reg

        def residual_ridge(params, A, b, regularization):
            """Невязка для L2-регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b
            if method == "leastsq":
                # Для leastsq возвращаем расширенный вектор невязок
                reg_residual = np.sqrt(regularization) * x
                return np.concatenate([residual, reg_residual])
            else:
                return np.sum(residual**2) + regularization * np.sum(x**2)

        def jacobian_ridge(params, A, b, regularization):
            """Якобиан для L2-регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            # Градиент: 2*A^T*(A*x - b) + 2*regularization*x
            return 2 * A.T @ (A @ x - b) + 2 * regularization * x

        def residual_elastic(params, A, b, regularization, regularization2, l1_weight):
            """Невязка для Elastic Net регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            residual = A @ x - b

            if method == "leastsq":
                # Для leastsq - комбинируем L1 и L2 регуляризацию
                l1_residual = (
                    np.sqrt(regularization * l1_weight) * np.sqrt(m) * np.abs(x)
                )
                l2_residual = np.sqrt(regularization2 * (1 - l1_weight)) * x
                reg_residual = np.concatenate([l1_residual, l2_residual])
                return np.concatenate([residual, reg_residual])
            else:
                # Скалярная целевая функция для других методов
                l1_penalty = regularization * l1_weight * np.sum(np.abs(x))
                l2_penalty = regularization2 * (1 - l1_weight) * np.sum(x**2)
                return np.sum(residual**2) + l1_penalty + l2_penalty

        def jacobian_elastic(params, A, b, regularization, regularization2, l1_weight):
            """Якобиан для Elastic Net регуляризации"""
            x = np.array([params[f"x{i}"].value for i in range(m)])
            # Градиент основной части
            grad_main = 2 * A.T @ (A @ x - b)
            # Градиент L1 части
            grad_l1 = regularization * l1_weight * np.sign(x)
            grad_l1[np.abs(x) < 1e-10] = 0  # Субградиент в точке 0
            # Градиент L2 части
            grad_l2 = 2 * regularization2 * (1 - l1_weight) * x
            return grad_main + grad_l1 + grad_l2

        not_supported_methods = [
            "emcee",
            "trust-krylov",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-constr",  # Jakobian and Hessian
            "brute",  # x0
            "ampgo",
            "dual_annealing",
        ]
        if method in not_supported_methods:
            print("Not supported")
            return None
        else:
            # Выбор модели и минимизация с учетом необходимости Якобиана
            gradient_methods = ["newton", "tnc", "cg", "bfgs", "lbfgsb"]

            if model_name == "lasso":
                if method in gradient_methods:
                    result = lmfit.minimize(
                        residual_lasso,
                        params,
                        args=(A, b, regularization),
                        method=method,
                        jac=jacobian_lasso,
                    )
                else:
                    result = lmfit.minimize(
                        residual_lasso,
                        params,
                        args=(A, b, regularization),
                        method=method,
                    )

            elif model_name == "ridge":
                if method in gradient_methods:
                    result = lmfit.minimize(
                        residual_ridge,
                        params,
                        args=(A, b, regularization),
                        method=method,
                        jac=jacobian_ridge,
                    )
                else:
                    result = lmfit.minimize(
                        residual_ridge,
                        params,
                        args=(A, b, regularization),
                        method=method,
                    )

            elif model_name == "elastic":
                if method in gradient_methods:
                    result = lmfit.minimize(
                        residual_elastic,
                        params,
                        args=(A, b, regularization, regularization2, l1_weight),
                        method=method,
                        jac=jacobian_elastic,
                    )
                else:
                    result = lmfit.minimize(
                        residual_elastic,
                        params,
                        args=(A, b, regularization, regularization2, l1_weight),
                        method=method,
                    )
            else:
                raise ValueError(
                    f"Неизвестная модель: {model_name}. Допустимые: 'elastic', 'lasso', 'ridge'"
                )

        # Извлекаем решение
        x = np.array([result.params[f"x{i}"].value for i in range(m)])
        computed_readings = A @ x
        residual = b - computed_readings
        # Базовая информация о результате
        result_dict = {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": residual,
            "residual norm": np.linalg.norm(residual),
            "solution norm": np.linalg.norm(x),
            "model_name": model_name,
            "solver": method,
            "method": f"lmfit, {method}",
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "chisqr": result.chisqr if hasattr(result, "chisqr") else None,
            "doserates": self._calculate_doserates(x),
        }

        # Расчет ошибок, если запрошено
        if calculate_errors and result.success:
            try:
                # Для некоторых методов lmfit может вычислить ковариационную матрицу
                if hasattr(result, "covar") and result.covar is not None:
                    errors = np.sqrt(np.diag(result.covar))
                    result_dict["errors"] = errors
                    result_dict["relative_errors"] = errors / (x + 1e-10)
            except Exception as e:
                result_dict["error_warning"] = f"Не удалось вычислить ошибки: {str(e)}"

        # Добавляем информацию о регуляризации
        result_dict.update(
            {
                "regularization": regularization,
                "regularization2": regularization2 if model_name == "elastic" else None,
                "l1_weight": l1_weight if model_name == "elastic" else None,
            }
        )

        return result_dict

    def unfold_tsvd(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        method: str = "discrepancy",
        k: int = None,
        threshold: float = None,
        noise_level: float = None,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Решение некорректной системы Ax = b с помощью усеченного SVD

        Параметры:
        A : numpy.ndarray [m x n]
        b : numpy.ndarray [m]
        k : int, количество сингулярных значений для удержания
        threshold : float, порог для отсечения малых сингулярных значений

        Возвращает:
        x : numpy.ndarray [n], решение
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        # Выполняем SVD разложение
        U, s, Vh = svd(A, full_matrices=False)
        V = Vh.T  # Транспонируем Vh для получения V

        # Определяем количество удерживаемых сингулярных значений
        if k is not None:
            # Используем заданное количество компонент
            k = min(k, len(s))
        elif threshold is not None:
            # Используем пороговое значение для отсечения
            k = np.sum(s / s[0] > threshold)
        else:
            # Автоматический выбор на основе убывания сингулярных чисел
            # k = len(s)

            def automatic_k_selection(s, A, b, method=method, noise_level=noise_level):
                """
                Автоматический выбор количества сингулярных значений для усечения

                Параметры:
                s : array, сингулярные значения
                A : array, матрица системы
                b : array, вектор правой части
                method : str, метод выбора ('discrepancy', 'l_curve', 'gcv', 'energy')
                noise_level : float, оценка уровня шума

                Возвращает:
                k : int, выбранное количество сингулярных значений
                """

                m, n = A.shape
                max_k = min(m, n)

                if method == "discrepancy":
                    # Метод невязки (Discrepancy Principle) - требует оценки уровня шума
                    if noise_level is None:
                        # Если уровень шума не задан, оцениваем его через наибольшее сингулярное число
                        noise_level = (
                            s[0] * 1e-3
                        )  # эвристика: 0.1% от максимального сингулярного числа

                    # Выполняем полное SVD для анализа
                    U, s_full, Vh = svd(A, full_matrices=False)

                    # Находим k, при котором норма невязки приближается к ожидаемому уровню шума
                    k = max_k
                    for i in range(1, max_k + 1):
                        # Вычисляем решение с i компонентами
                        s_i = s_full[:i]
                        U_i = U[:, :i]
                        V_i = Vh[:i, :].T
                        x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b

                        # Вычисляем невязку
                        residual = np.linalg.norm(A @ x_i - b)

                        # Проверяем условие невязки
                        if residual <= noise_level * np.sqrt(max(m - i, 1)):
                            k = i
                            break

                    print(
                        f"Метод невязки: выбрано k={k} (уровень шума={noise_level:.2e})"
                    )
                    return k

                elif method == "energy":
                    # Метод накопленной энергии (вариант 1: порог по энергии)
                    energy_threshold = 0.95  # сохраняем 95% энергии
                    cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
                    k = np.argmax(cumulative_energy >= energy_threshold) + 1

                    # Вариант 2: перегиб на графике энергии (скачок)
                    # diff_energy = np.diff(cumulative_energy)
                    # k = np.argmax(diff_energy < 0.01) + 1  # когда прирост становится меньше 1%

                    print(
                        f"Метод энергии: выбрано k={k} ({cumulative_energy[k - 1]:.1%} энергии)"
                    )
                    return k

                elif method == "l_curve":
                    # Метод L-образной кривой (находим "угол")
                    U, s_full, Vh = svd(A, full_matrices=False)
                    V = Vh.T

                    residual_norms = []
                    solution_norms = []

                    # Вычисляем кривую для всех возможных k
                    for i in range(1, min(len(s_full), n) + 1):
                        s_i = s_full[:i]
                        U_i = U[:, :i]
                        V_i = V[:, :i]
                        x_i = V_i @ np.diag(1.0 / s_i) @ U_i.T @ b

                        residual_norms.append(np.linalg.norm(A @ x_i - b))
                        solution_norms.append(np.linalg.norm(x_i))

                    # Преобразуем в логарифмическую шкалу
                    log_res = np.log(residual_norms)
                    log_sol = np.log(solution_norms)

                    # Вычисляем кривизну
                    curvature = []
                    for i in range(1, len(log_res) - 1):
                        # Аппроксимируем производные
                        dx1 = log_res[i] - log_res[i - 1]
                        dy1 = log_sol[i] - log_sol[i - 1]
                        dx2 = log_res[i + 1] - log_res[i]
                        dy2 = log_sol[i + 1] - log_sol[i]

                        # Кривизна
                        curv = abs(dx1 * dy2 - dx2 * dy1) / (
                            (dx1**2 + dy1**2) ** 1.5 + 1e-10
                        )
                        curvature.append(curv)

                    # Находим точку максимальной кривизны (угол L-кривой)
                    if len(curvature) > 0:
                        k_idx = np.argmax(curvature) + 1  # +1 из-за смещения индексов
                        k = min(k_idx + 1, len(s))  # +1 для компенсации
                    else:
                        k = len(s) // 2  # эвристика по умолчанию

                    print(f"Метод L-кривой: выбрано k={k}")
                    return k

                elif method == "gcv":
                    # Generalized Cross-Validation (GCV)
                    U, s_full, Vh = svd(A, full_matrices=False)

                    # Вычисляем коэффициенты проекции b на левые сингулярные векторы
                    beta = U.T @ b

                    gcv_values = []
                    k_values = range(1, min(len(s_full), n) + 1)

                    for i in k_values:
                        # Остаточные компоненты
                        residual = np.sum(beta[i:] ** 2)
                        # Эффективное число параметров
                        eff_params = m - i

                        # Функция GCV
                        if eff_params > 0:
                            gcv = residual / (eff_params**2)
                        else:
                            gcv = np.inf

                        gcv_values.append(gcv)

                    # Находим минимум GCV
                    k = k_values[np.argmin(gcv_values)]
                    print(
                        f"Метод GCV: выбрано k={k} (min GCV={np.min(gcv_values):.2e})"
                    )
                    return k

                elif method == "threshold_ratio":
                    # Метод порога по отношению сингулярных чисел
                    threshold_ratio = (
                        1e-2  # порог 1% от максимального сингулярного числа
                    )
                    s_normalized = s / s[0]  # нормализуем по максимальному значению
                    k = np.sum(s_normalized > threshold_ratio)

                    print(f"Метод порога: выбрано k={k} (порог={threshold_ratio})")
                    return k

                elif method == "median_threshold":
                    # Метод медианного порога (более устойчивый)
                    median_s = np.median(s)
                    k = np.sum(s >= median_s)

                    print(f"Медианный порог: выбрано k={k} (медиана={median_s:.2e})")
                    return k

                elif method == "donoho":
                    # решение с подбором r_cond из статьи Donoho.
                    # arXiv:1305.5870v1 [stat.ME] 24 May 2013
                    # The Optimal Hard Threshold for Singular Values
                    # is 4/√3 David L. Donoho ∗ Matan Gavish ∗

                    U, s_full, Vh = svd(A, full_matrices=False)

                    sigma_donoho = 0.05  # 5% error
                    donoho_rcond = 4 / np.sqrt(3) * np.sqrt(n) * sigma_donoho
                    print(f"Donoho_rcond = {donoho_rcond}")
                    # Сумма булевых значений (True=1, False=0)
                    k = np.sum(s_full > donoho_rcond)
                    print(f"{k=}")
                    return k

                else:
                    # Эвристика по умолчанию: удерживаем сингулярные числа, которые больше среднего
                    mean_s = np.mean(s)
                    k = np.sum(s >= mean_s)

                    print(
                        f"Эвристика по среднему: выбрано k={k} (среднее={mean_s:.2e})"
                    )
                    return k

            # автоматически подобрали k
            k = automatic_k_selection(s, A, b, method=method, noise_level=noise_level)

        print(f"Используется {k} сингулярных значений из {len(s)}")
        print(f"Сингулярные значения: {s[:10]}...")  # Показываем первые 10

        # Обрезаем сингулярные значения и матрицы
        s_k = s[:k]
        U_k = U[:, :k]
        V_k = V[:, :k]

        # Вычисляем псевдообратную матрицу с усечением
        # Σ⁺ = diag(1/s₁, 1/s₂, ..., 1/sₖ, 0, ..., 0)
        Sigma_pinv = np.zeros((A.shape[1], A.shape[0]))
        Sigma_pinv[:k, :k] = np.diag(1.0 / s_k)

        # Вычисляем усеченное псевдообратное преобразование Мура-Пенроуза
        A_pinv = V_k @ Sigma_pinv[:k, :k] @ U_k.T

        # Вычисляем решение
        x = A_pinv @ b

        computed_readings = A @ x
        residual = computed_readings - b
        residual_norm = np.linalg.norm(residual)

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "effective_readings": dict(zip(selected, computed_readings)),
            "residual": residual,
            "residual norm": residual_norm,
            "k": k,
            "threshold": threshold,
            "find_k_kmethod": method,
            "doserates": self._calculate_doserates(x),
            "method": "TSVD",
        }

    def unfold_qubo(
        self,
        readings: Dict[str, float],
        initial_spectrum: Optional[np.ndarray] = None,
        regularization: float = 1e-4,
        num_reads: int = 200,
        calculate_errors: bool = False,
    ) -> Dict:
        """
        Восстановление спектра с помощью алгоритма Qunfold.
        Quadratic unconstrained binary optimization (QUBO)
        This package provides a quantum-based approach to the statistical unfolding problem
        in High-Energy Physics (HEP). The technique is based on the reformulation
        of this task as a Quadratic Unconstrained Binary Optimization (QUBO) problem
        to be solved by Quantum Annealing (QA) on D-Wave quantum devices.

        Параметры:
        readings : Dict[str, float]
            Показания детекторов
        initial_spectrum : Optional[np.ndarray], optional
            Начальное приближение спектра. Если None, используется равномерный спектр.
        regularization : float, optional
            Параметр регуляризации. По умолчанию 1e-4
        calculate_errors : bool, optional
            Флаг расчета ошибок восстановления. По умолчанию False

        Возвращает:
        Dict
            Словарь с результатами восстановления спектра.
        """
        # Валидация и подготовка данных
        readings = self._validate_readings(readings)
        selected = [name for name in self.detector_names if name in readings]

        # Векторизованная подготовка данных
        b = np.array([readings[name] for name in selected], dtype=float)
        A = np.array([self.sensitivities[name] for name in selected], dtype=float)

        m, n = A.shape

        # границ бинов должно быть на 1 больше чем самих бинов
        binning = np.append(np.array(self.E_MeV), np.array(np.max((self.E_MeV))))

        unfolder = QUnfolder(A.T @ A, A.T @ b, binning, lam=regularization)
        unfolder.initialize_qubo_model()
        x, cov = unfolder.solve_simulated_annealing(num_reads=num_reads)
        # Формирование результатов
        computed_readings = A @ x
        residual = computed_readings - b
        # print(f"\nНорма невязки: {np.linalg.norm():.6f}")

        return {
            "energy": self.E_MeV.copy(),
            "spectrum": x,
            "spectrum_absolute": x,
            "spectrum error": np.sqrt(np.diag(cov)),
            "covariation": cov,
            "residual": residual,
            "residual norm": np.linalg.norm(residual),
            "effective_readings": dict(zip(selected, computed_readings)),
            "regularization": regularization,
            "num reads": num_reads,
            "doserates": self._calculate_doserates(x),
            "method": "Quadratic unconstrained binary optimization (QUBO)",
        }

    def _calculate_doserates(self, spectrum, dlnE=0.2):
        """
        расчёт доз по ICRP-116
        равномерный шаг в логарифме 0.2
        spectrum: array grid
        :param dlnE: Description
        """
        if not self.cc_icrp116:  # или len(self.cc_icrp116) == 0
            return {geom: 0.0 for geom in ["AP", "PA", "LLAT", "RLAT", "ISO", "ROT"]}

        doserates = {}
        for geom, k in self.cc_icrp116.items():
            if geom != "E_MeV":
                integrand = k * spectrum * dlnE
                dose = np.log(10) * np.sum(integrand)
                doserates[geom] = float(dose)  # pcSv/s
        return doserates

    def _load_icrp116_coefficients(self):
        """Load ICRP-116 conversion coefficients"""
        # ICRP-116 conversion coefficients (from the provided table)
        data = ICRP116_COEFF_EFFECTIVE_DOSE
        return data

    def _add_noise(self, readings, noise_level=0.01):
        """
        Добавление нормального шума к каждому значению в словаре

        Parameters:
        -----------
        readings_dict : dict
            Словарь с исходными данными
        noise_level : float
            Уровень шума в процентах (по умолчанию 1%)

        Returns:
        --------
        dict : Словарь с шумом
        """
        readings_noisy = {}
        for key, value in readings.items():
            # Генерация нормального шума
            noise = np.random.normal(loc=0, scale=noise_level)
            # Применение шума
            readings_noisy[key] = value * (1 + noise)
        return readings_noisy

    # def get_uncertainty(self,n_iterations=500,noise_level=0.01)
