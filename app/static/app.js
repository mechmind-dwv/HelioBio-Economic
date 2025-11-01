// 🌞 HelioBio-Economic Dashboard - JavaScript Principal
class HelioBioDashboard {
    constructor() {
        this.apiBaseUrl = window.location.origin + '/api';
        this.charts = {};
        this.updateInterval = 30000; // 30 segundos
        this.init();
    }

    init() {
        console.log('🚀 Inicializando HelioBio-Economic Dashboard');
        this.checkSystemStatus();
        this.loadDashboardData();
        this.setupEventListeners();
        this.startAutoRefresh();
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/system/health`);
            const data = await response.json();
            
            const statusElement = document.getElementById('systemStatus');
            if (data.success) {
                statusElement.className = 'status-badge status-healthy';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Sistema Conectado';
            } else {
                statusElement.className = 'status-badge status-error';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Sistema con Errores';
            }
        } catch (error) {
            console.error('Error checking system status:', error);
            document.getElementById('systemStatus').className = 'status-badge status-error';
            document.getElementById('systemStatus').innerHTML = '<i class="fas fa-circle"></i> Error de Conexión';
        }
    }

    async loadDashboardData() {
        await Promise.all([
            this.loadSolarData(),
            this.loadEconomicData(),
            this.loadCorrelationData(),
            this.loadSystemHealth(),
            this.loadKondratievAnalysis(),
            this.loadRecentEvents()
        ]);
    }

    async loadSolarData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/solar/current`);
            const data = await response.json();

            if (data.success) {
                const solar = data.data.solar_activity;
                
                // Actualizar métricas
                document.getElementById('sunspotNumber').textContent = solar.sunspot_number;
                
                // Actualizar tendencia
                const trendElement = document.getElementById('solarTrend');
                if (solar.sunspot_number > 50) {
                    trendElement.className = 'trend up';
                    trendElement.innerHTML = '<i class="fas fa-arrow-up"></i> <span>Alta Actividad</span>';
                } else {
                    trendElement.className = 'trend down';
                    trendElement.innerHTML = '<i class="fas fa-arrow-down"></i> <span>Baja Actividad</span>';
                }

                // Crear/actualizar gráfico
                this.updateSolarChart(solar);
            }
        } catch (error) {
            console.error('Error loading solar data:', error);
        }
    }

    async loadEconomicData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/economic/markets?symbol=^GSPC&period=1mo`);
            const data = await response.json();

            if (data.success) {
                const marketData = data.data;
                
                // Actualizar métricas
                document.getElementById('sp500Value').textContent = marketData.current_price.toFixed(2);
                
                // Calcular tendencia
                const prices = marketData.market_data.map(item => item.price);
                const currentPrice = prices[prices.length - 1];
                const previousPrice = prices[prices.length - 2];
                const trend = ((currentPrice - previousPrice) / previousPrice * 100);

                const trendElement = document.getElementById('economicTrend');
                if (trend > 0) {
                    trendElement.className = 'trend up';
                    trendElement.innerHTML = `<i class="fas fa-arrow-up"></i> <span>+${trend.toFixed(2)}%</span>`;
                } else {
                    trendElement.className = 'trend down';
                    trendElement.innerHTML = `<i class="fas fa-arrow-down"></i> <span>${trend.toFixed(2)}%</span>`;
                }

                // Crear/actualizar gráfico
                this.updateEconomicChart(marketData);
            }
        } catch (error) {
            console.error('Error loading economic data:', error);
        }
    }

    async loadCorrelationData() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/correlation/solar-economic?economic_indicator=SP500&solar_indicator=sunspots&period_years=10`);
            const data = await response.json();

            if (data.success) {
                const correlation = data.data.correlation_results;
                
                // Actualizar métricas
                const corrValue = correlation.methods.pearson;
                document.getElementById('correlationValue').textContent = corrValue.toFixed(3);
                
                // Color basado en fuerza de correlación
                const corrElement = document.getElementById('correlationValue');
                if (Math.abs(corrValue) > 0.7) {
                    corrElement.className = 'metric correlation high-correlation';
                } else if (Math.abs(corrValue) > 0.4) {
                    corrElement.className = 'metric correlation medium-correlation';
                } else {
                    corrElement.className = 'metric correlation low-correlation';
                }

                // Actualizar tendencia
                document.getElementById('correlationTrend').innerHTML = 
                    `<i class="fas fa-info-circle"></i> <span>${correlation.significance}</span>`;

                // Crear/actualizar gráfico
                this.updateCorrelationChart(correlation);
            }
        } catch (error) {
            console.error('Error loading correlation data:', error);
        }
    }

    async loadSystemHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/system/health`);
            const data = await response.json();

            if (data.success) {
                const health = data.data;
                const html = `
                    <div style="display: grid; gap: 10px;">
                        <div style="display: flex; justify-content: between; align-items: center;">
                            <span>NASA API:</span>
                            <span class="status-badge ${health.services.nasa_solar_service.overall_status === 'healthy' ? 'status-healthy' : 'status-error'}">
                                ${health.services.nasa_solar_service.overall_status}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: between; align-items: center;">
                            <span>Datos Económicos:</span>
                            <span class="status-badge ${health.services.economic_data_service.overall_status === 'healthy' ? 'status-healthy' : 'status-error'}">
                                ${health.services.economic_data_service.overall_status}
                            </span>
                        </div>
                        <div style="display: flex; justify-content: between; align-items: center;">
                            <span>CPU:</span>
                            <span>${health.system_metrics.cpu_percent}%</span>
                        </div>
                        <div style="display: flex; justify-content: between; align-items: center;">
                            <span>Memoria:</span>
                            <span>${health.system_metrics.memory_usage}%</span>
                        </div>
                    </div>
                `;
                document.getElementById('systemHealth').innerHTML = html;
            }
        } catch (error) {
            console.error('Error loading system health:', error);
        }
    }

    async loadKondratievAnalysis() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/economic/kondratiev`);
            const data = await response.json();

            if (data.success) {
                const analysis = data.data.current_analysis;
                const html = `
                    <div style="text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 1rem;">
                            Onda ${analysis.current_wave} - ${analysis.current_phase}
                        </div>
                        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;">
                            <div>Progreso: ${(analysis.phase_progress * 100).toFixed(1)}%</div>
                            <div style="height: 10px; background: rgba(255,255,255,0.2); border-radius: 5px; margin: 0.5rem 0;">
                                <div style="height: 100%; background: var(--success); border-radius: 5px; width: ${analysis.phase_progress * 100}%"></div>
                            </div>
                            <div style="font-size: 0.9rem; opacity: 0.8;">
                                Próxima transición: ${new Date(analysis.next_phase_transition).toLocaleDateString()}
                            </div>
                        </div>
                    </div>
                `;
                document.getElementById('kondratievInfo').innerHTML = html;
            }
        } catch (error) {
            console.error('Error loading Kondratiev analysis:', error);
        }
    }

    async loadRecentEvents() {
        try {
            const [solarResponse, economicResponse] = await Promise.all([
                fetch(`${this.apiBaseUrl}/solar/flares?days=3`),
                fetch(`${this.apiBaseUrl}/economic/conditions`)
            ]);

            const solarData = await solarResponse.json();
            const economicData = await economicResponse.json();

            let events = [];

            // Eventos solares
            if (solarData.success && solarData.data.flares.length > 0) {
                solarData.data.flares.slice(0, 3).forEach(flare => {
                    events.push({
                        type: 'solar',
                        message: `Fulguración ${flare.class_type}`,
                        time: new Date(flare.peak_time).toLocaleDateString(),
                        severity: flare.class_type === 'X' ? 'high' : flare.class_type === 'M' ? 'medium' : 'low'
                    });
                });
            }

            // Eventos económicos
            if (economicData.success) {
                events.push({
                    type: 'economic',
                    message: `Mercado: ${economicData.data.market_condition}`,
                    time: 'Actual',
                    severity: economicData.data.market_condition.includes('bull') ? 'low' : 'medium'
                });
            }

            const html = events.map(event => `
                <div style="display: flex; align-items: center; gap: 10px; padding: 0.5rem; background: rgba(255,255,255,0.05); border-radius: 8px; margin-bottom: 0.5rem;">
                    <i class="fas fa-${event.type === 'solar' ? 'sun' : 'chart-line'}" 
                       style="color: ${event.type === 'solar' ? 'var(--solar)' : 'var(--economic)'}"></i>
                    <div style="flex: 1;">
                        <div style="font-weight: 600;">${event.message}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">${event.time}</div>
                    </div>
                    <div class="status-badge ${event.severity === 'high' ? 'status-error' : event.severity === 'medium' ? 'status-warning' : 'status-healthy'}">
                        ${event.severity}
                    </div>
                </div>
            `).join('');

            document.getElementById('recentEvents').innerHTML = html || '<p>No hay eventos recientes</p>';
        } catch (error) {
            console.error('Error loading recent events:', error);
        }
    }

    updateSolarChart(solarData) {
        const ctx = document.getElementById('solarActivityChart').getContext('2d');
        
        if (this.charts.solar) {
            this.charts.solar.destroy();
        }

        this.charts.solar = new Chart(ctx, {
            type: 'line',
            data: {
                labels: ['Manchas', 'Flujo Solar', 'Índice Kp', 'Viento Solar'],
                datasets: [{
                    label: 'Actividad Solar',
                    data: [
                        solarData.sunspot_number / 2, // Normalizar
                        solarData.solar_flux / 3,     // Normalizar
                        solarData.kp_index * 10,      // Escalar
                        solarData.wind_speed / 10     // Normalizar
                    ],
                    backgroundColor: 'rgba(251, 191, 36, 0.2)',
                    borderColor: 'rgba(251, 191, 36, 1)',
                    borderWidth: 2,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        }
                    }
                }
            }
        });
    }

    updateEconomicChart(marketData) {
        const ctx = document.getElementById('economicChart').getContext('2d');
        const prices = marketData.market_data.map(item => item.price);
        const dates = marketData.market_data.map(item => 
            new Date(item.timestamp).toLocaleDateString()
        );

        if (this.charts.economic) {
            this.charts.economic.destroy();
        }

        this.charts.economic = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dates,
                datasets: [{
                    label: 'S&P 500',
                    data: prices,
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    borderColor: 'rgba(59, 130, 246, 1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)',
                            maxTicksLimit: 6
                        }
                    }
                }
            }
        });
    }

    updateCorrelationChart(correlationData) {
        const ctx = document.getElementById('correlationChart').getContext('2d');
        const methods = Object.keys(correlationData.methods);
        const values = Object.values(correlationData.methods);

        if (this.charts.correlation) {
            this.charts.correlation.destroy();
        }

        this.charts.correlation = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: methods.map(m => m.charAt(0).toUpperCase() + m.slice(1)),
                datasets: [{
                    label: 'Coeficiente',
                    data: values,
                    backgroundColor: values.map(v => 
                        Math.abs(v) > 0.7 ? 'rgba(16, 185, 129, 0.8)' :
                        Math.abs(v) > 0.4 ? 'rgba(245, 158, 11, 0.8)' :
                        'rgba(239, 68, 68, 0.8)'
                    ),
                    borderColor: values.map(v => 
                        Math.abs(v) > 0.7 ? 'rgba(16, 185, 129, 1)' :
                        Math.abs(v) > 0.4 ? 'rgba(245, 158, 11, 1)' :
                        'rgba(239, 68, 68, 1)'
                    ),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        min: -1,
                        max: 1,
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255,255,255,0.1)'
                        },
                        ticks: {
                            color: 'rgba(255,255,255,0.7)'
                        }
                    }
                }
            }
        });
    }

    setupEventListeners() {
        // Los listeners de pestañas ya están en el HTML
        console.log('Event listeners configurados');
    }

    startAutoRefresh() {
        setInterval(() => {
            this.loadDashboardData();
            this.checkSystemStatus();
        }, this.updateInterval);
    }
}

// Funciones globales para las pestañas
function showTab(tabName) {
    // Ocultar todas las pestañas
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Mostrar la pestaña seleccionada
    document.getElementById(tabName).classList.add('active');
    
    // Actualizar botones de navegación
    document.querySelectorAll('.nav-tab').forEach(button => {
        button.classList.remove('active');
    });
    event.target.classList.add('active');

    // Cargar datos específicos de la pestaña
    switch(tabName) {
        case 'solar':
            dashboard.loadSolarData();
            break;
        case 'economic':
            dashboard.loadEconomicData();
            break;
        case 'correlation':
            dashboard.loadCorrelationData();
            break;
        case 'predictions':
            // dashboard.loadPredictionsData();
            break;
    }
}

// Inicializar dashboard cuando se carga la página
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new HelioBioDashboard();
});
