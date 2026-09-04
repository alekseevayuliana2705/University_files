alpha = 0.1; 
beta = 10;
t = 0.9;
% флаг для отображения траекторий (0 - не отображать)
flag = 0; 
[X, Y, X1, Y1, X2, T2] = reachset(alpha, beta, t, flag);
hold on;
plot(X, Y, 'm', 'LineWidth', 2); % Граница множества достижимости
plot(X1, Y1, 'b', 'LineWidth', 2); % Линии переключения управления
plot(0, 0, 'k*', 'LineWidth', 2); % Начальная точка
xlabel('$x_1$', 'Interpreter', 'Latex', 'FontSize', 10);
ylabel('$x_2$', 'Interpreter', 'Latex', 'FontSize', 10);
grid on;

t1 = 0.01;
t2 = t;
N = 10;

% Для анимации можно раскомментировать:
% reachsetdyn(alpha, beta, t1, t2, N, flag);
% reachsetdyn(alpha, beta, t1, t2, N, flag, 'lab3');
function [X,Y,X1,Y1,X2,T2] = reachset(alpha, beta, t, flag)
    n = 100; % количество шагов
    % Координаты точек траекторий системы
    arr1 = []; % для подсистемы S+
    arr2 = []; % для подсистемы S-
    % Настройки решателя ОДУ
    x_opts = odeset('Events', @event_x);
    psi_opts = odeset('Events', @event_psi);
    % Поиск стационарных точек для обеих подсистем
    [stat_plus, found_plus] = find_stationary_points(alpha, beta, 1); % Для S+ (u=alpha)
    [stat_minus, found_minus] = find_stationary_points(alpha, beta, 2); % Для S- (u=-alpha)
    if found_plus
        X2 = stat_plus;
        T2 = ones(size(stat_plus,1),1); % Номер подсистемы 1
    end
    if found_minus
        X2 = [X2; stat_minus];
        T2 = [T2; 2*ones(size(stat_minus,1),1)]; % Номер подсистемы 2
    end
    % Решение первой системы (предполагаем psi_2 > 0 вначале)
    [time, x_plus] = ode45(@(t,x) sys1(t,x,alpha, beta),[0 t], [0 0], x_opts);
    % Координаты точек переключения
    switch_x = x_plus;
    % Интерполяция
    x_plus_func = @(t) interp1(time, x_plus, t, 'spline');
    tm = linspace(0, time(end), n);
    for i = 1:n
        t_cur = tm(i);
        x_cur = x_plus_func(tm(1:i));
        while (t_cur < t)
            % Решаем S-
            [ode_t, ode_x] = ode45(@(t,x) ssys2(t, x, alpha, beta), [t_cur, t], ...
                [x_cur(end, 1), x_cur(end, 2), 1, 0], psi_opts);
            x_cur = [x_cur; ode_x(2:end, 1:2)];
            arr2 = [arr2; ode_x(2:end, 1:2)];
            % Запись координат точек пересечения
            if (ode_t(end) >= t)
                break
            else
                switch_x = [switch_x; ode_x(end, 1:2)];
            end
            % Решаем S+
            [ode_t, ode_x] = ode45(@(t,x) ssys1(t,x,alpha, beta), [ode_t(end), t],...
                [x_cur(end, 1), x_cur(end, 2), -1, 0], psi_opts);
            x_cur = [x_cur; ode_x(2:end, 1:2)];
            arr1 = [arr1; ode_x(2:end, 1:2)];
            % Обновление времени
            t_cur = ode_t(end);
            % Запись координат точек пересечения
            if (ode_t(end) < t)
                switch_x = [switch_x; ode_x(end, 1:2)];
            end
        end
        % Отрисовка траекторий
        if (flag)
            plot(x_cur(:, 1), x_cur(:, 2), 'c');        
        end
    end 
     % Решение второй системы (предполагаем psi_2 < 0 вначале)
    [time, x_minus] = ode45(@(t,x) sys2(t,x, alpha, beta),[0 t], [0 0], x_opts);
    % Интерполяция
    x_minus_func = @(t) interp1(time, x_minus, t, 'spline');
    % Координаты точек переключения
    switch_x = [flip(x_minus);switch_x];
    tm = linspace(0, time(end), n);
    for i=1:n
        t_cur = tm(i);
        x_cur = x_minus_func(tm(1:i));
        while (t_cur < t)
            % Решаем S+
            [ode_t, ode_x] = ode45(@(t,x) ssys1(t,x,alpha, beta), [t_cur, t],...
                [x_cur(end, 1), x_cur(end, 2), -1, 0], psi_opts);
            x_cur = [x_cur; ode_x(2:end, 1:2)];
            arr1 = [arr1; ode_x(2:end, 1:2)];
            if (ode_t(end) >= t)
                break
            else
                switch_x = [ode_x(end, 1:2); switch_x];
            end
            % Решаем S-
            [ode_t, ode_x] = ode45(@(t,x) ssys2(t,x,alpha, beta), [ode_t(end), t],...
                [x_cur(end, 1), x_cur(end, 2), 1, 0], psi_opts);
            x_cur = [x_cur; ode_x(2:end, 1:2)];
            arr2 = [arr2; ode_x(2:end, 1:2)];
            % Обновление времени
            t_cur = ode_t(end);
            % Запись координат точек пересечения
            if (ode_t(end) < t)
                switch_x = [ode_x(end, 1:2); switch_x];
            end
        end
        % Отрисовка траекторий
        if (flag)
            plot(x_cur(:, 1), x_cur(:, 2), 'c');
        end
    end
    % Объединение всех кривых
    unity = [arr1; arr2];
    % Выделение границы
    k = boundary(unity(:, 1), unity(:, 2), 0.1);
    % Координаты границы множества достижимости
    X = unity(k, 1);
    Y = unity(k, 2);
    % Координаты линий переключения оптимального управления
    [X1, k] = sort(switch_x(:, 1));
    Y1 = switch_x(k, 2);
end

% Событие для решателя - пересечение нуля psi_2
function [value, isterminal, direction] = event_psi(~, x)
    value = x(4); % psi_2
    isterminal = 1;
    direction = 0;
end

% Событие для решателя - пересечение нуля x_2
function [value, isterminal, direction] = event_x(~, x)
    value = x(2); % x_2
    isterminal = 1;
    direction = 0;
end

% Исходная система с u = alpha
function func = sys1(~,x,alpha, beta)
    func = [x(2); -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) + alpha];
end

% Исходная система с u = -alpha
function func = sys2(~,x,alpha, beta)
    func = [x(2); -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) - alpha];
end

% Сопряженная система S+
function func = ssys1(~,x,alpha, beta)
    func = [x(2);
        -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) + alpha;
         beta*x(4) - 18*x(1)^2*x(4)*cos(3*x(1)^3) + x(4)*x(2);
        -x(3) + x(4)*x(1)];
end

% Сопряженная система S-
function func = ssys2(~,x,alpha, beta)
    func = [x(2);
        -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) - alpha;
        beta*x(4) - 18*x(1)^2*x(4)*cos(3*x(1)^3) + x(4)*x(2);
        -x(3) + x(4)*x(1)];
end

% Поиск стационарных точек
function [stat_points, found] = find_stationary_points(alpha, beta, system_num)
    stat_points = [];
    found = false;

    % Уравнения для стационарных точек: dx1/dt = 0, dx2/dt = 0
    if system_num == 1
        fun = @(x) [x(2); -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) + alpha];
    else 
        fun = @(x) [x(2); -beta*x(1) + 2*sin(3*x(1)^3) - x(1)*x(2) - alpha];
    end

    % Начальные приближения для поиска
    initial_guesses = [0 0; 1 1; -1 1; 1 -1; -1 -1];
    options = optimset('Display','off');
    for k = 1:size(initial_guesses,1)
        % Решение нелинейной системы уравнений
        [x_sol, ~, exitflag] = fsolve(fun, initial_guesses(k,:), options);
         
        if exitflag > 0 % Решение найдено
            % Проверка на дубликаты
            if isempty(stat_points)
                stat_points = x_sol;
                found = true;
            else
                distances = sqrt(sum((stat_points - x_sol).^2, 2));
                if all(distances >= 1e-6)
                    stat_points = [stat_points; x_sol];
                    found = true;
                end
            end
        end
    end
end

% Функция для анимации изменения множества достижимости
function reachsetdyn(alpha, beta, t1, t2, N, flag, filename)
    mov(N) = struct('cdata', [], 'colormap', []);
    % Границы для отображения
    [X_max, Y_max] = reachset(alpha, beta, t2, flag);
    axis([min(X_max), max(X_max), min(Y_max), max(Y_max)]);
    % Сохранение видео или показ анимации
    if t1 ~= t2
        for i = 0:N
            tau = t1 + (t2 - t1)*i/N;
            [X, Y] = reachset(alpha, beta, tau, flag);
            plot(X, Y, 'm', 'LineWidth', 1);
            mov(i + 1) = getframe;
        end
        if nargin == 7
            vidObj = VideoWriter(strcat(filename, '.avi')); 
            vidObj.FrameRate = 5; 
            open(vidObj);
            writeVideo(vidObj,mov);
            close(vidObj);
        else                   
            movie(mov, 1, 60);
        end
    else
         [X, Y] = reachset(alpha, beta, t1, flag);
         plot(X, Y, 'm', 'LineWidth', 1);
    end
end