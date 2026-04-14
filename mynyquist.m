% mynyquist

function y=mynyquist(tf,fignum)

    figure(fignum)
    nyqlog(tf)
    set(gca,'fontsize',20)
    lines = findobj(gca,'type','line');
    lines(51).Color = 'blue';
    lines(51).LineWidth = 7;
    lines(52).LineWidth = 7;

    text = findobj(gca,'type','text');
    num_texts = size(text,1);
    for i = 1:num_texts
        text(i).FontSize = 20;
    end
end