function [plotter,trainAccPlotter,valAccPlotter] = initializeTrainingProgressPlot()
figure% Plot the loss, training accuracy, and validation accuracy.
subplot(2,1,1)% Loss plot
plotter = animatedline('Color','b','linewidth',2);
xlabel("Iteration");ylabel("Loss")
subplot(2,1,2);% Accuracy plot
trainAccPlotter = animatedline('Color','r','linewidth',2);
valAccPlotter = animatedline('Color','b','linewidth',2);
legend('Training Accuracy','Validation Accuracy','Location','northwest');
xlabel("Iteration");ylabel("Accuracy")
end