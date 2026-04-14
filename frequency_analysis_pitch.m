clear all
close all

% This is the integral + LQR control setup for the linearization about 200
% m/s, 0 climb angle.
% close all
load('Output/MATLAB_files/allmodels.mat')

models_A = ["A_initial","A_volmax","A_110max","A_volmax_110con"];
models_B = ["B_initial","B_volmax","B_110max","B_volmax_110con"];


% models_A = ["A_initial","A_volmax_110con"];
% models_B = ["B_initial","B_volmax_110con"];

for i = 1:length(models_A)
    A = eval(models_A(i));
    B = eval(models_B(i));

    model_string = models_A(i);
    model_name = model_string{1}(3:end);
    % fprintf('%s model\n',model_name)

    B1 = B(:,1);
    B2 = B(:,2);
    
    C = [0 0 0 1];
    Abar = [A zeros(4,1);C 0];
    Bbar = [B;zeros(1,2)];
    Cbar = [C 0];
    
    D = 0;
    
    % Weightings for pitch integrator control
    % These values are from the nonlinear sims in the paper
    % Q = diag([1,100,1,100,100]);
    % R = diag([0.1,1000]);

    Q = diag([1,100,1,100,1000]);
    R = diag([0.1,1000]);
   
    
    Kbar = lqr(Abar,Bbar,Q,R);
    
    % Modify matrices to close loops on all signals except pitch and thrust
    Ktheta = Kbar(2,end-1);
    Ksigma = Kbar(2,end);
    % Kprime = Kbar(:,1:4);
    % Kprime(1,1) = 0;
    
    K = tf([Ktheta Ksigma],[1 0]);
    K2 = K * tf([10],[1 10]);
    
    %% Form loop TF
    
    Bbar1 = Bbar(:,1);
    Bbar2 = Bbar(:,2);
    
    Kbarprime = Kbar;
    Kbarprime(2,end-1) = 0;
    Kbarprime(2,end) = 0;
    
    Gbar = ss(Abar-Bbar*Kbarprime, Bbar2, Cbar, D);
    Gbar = tf(Gbar);
    
    Lbar = K2*Gbar;
    Sbar = 1/(1+Lbar);
    
    %% Plot
    
    % basefig = (i-1)*10;
    % figure(basefig+1)
    % bode(Lbar)
    % 
    % figure(basefig+2)
    % mynyquist(Lbar,basefig+2)
    % title("Nyquist")
    % print('-dpng','-r300',sprintf('Output/MATLAB_files/%s_nyq',model_name))
    % 
    % figure(basefig+3)
    % loops = loopsens(Gbar,K);
    % bode(loops.Si,'r',loops.Ti,'b',loops.Li,'g--')
    % title('Method 2')
    % xlim([10^-2,10^2])
    % legend('Sensitivity','Complementary Sensitivity','Loop Transfer')
    
    %% Gain Crossover
    
    [Gm,Pm,Wcg,Wcp] = margin(Lbar);
    fprintf('model: %s \n',model_name)
    fprintf('Gain crossover frequency is %1.3f rad/s\n',Wcp)
    fprintf('Gain Margin: %1.3f dB \n',20*log10(Gm))


    %% Overlay all Bode responses

    figure(100)
    bode(Lbar)
    hold on

    figure(200)
    bode(1/(Lbar+1))
    hold on

end

figure(100)
grid on
lines = findall(gcf,'type','line');
% Pull out even indices to make lines bigger
for i = 1:floor(length(lines)/2)
    lines(2*i).LineWidth = 5;
end
xlim([10^-4,10^2])

[~, hobj, ~, ~] = legend({models_A{1}(3:end),models_A{2}(3:end),models_A{3}(3:end),models_A{4}(3:end)},'Location','Northeast');
hl = findobj(hobj,'type','line');
set(hl,'LineWidth',5);
ht = findobj(hobj,'type','text');
set(ht,'FontSize',20);


% legend(models_A{1}(3:end),models_A{2}(3:end),models_A{3}(3:end),models_A{4}(3:end))
title('Bode of L(s) for Initial and Optimized Designs')
% legend('Inital','Optimized')


figure(200)
grid on
xlim([10^-4,10^2])
lines = findall(gcf,'type','line');
for i = 1:floor(length(lines)/2)
    lines(2*i).LineWidth = 5;
end
legend(models_A{1}(3:end),models_A{2}(3:end),models_A{3}(3:end),models_A{4}(3:end))
% legend("Initial","Volmax","110Max","Volmax_110con")
title('Bode of S(s) for Initial and Optimized Designs')