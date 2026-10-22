% Example: read results1.txt written by preview_file.py.
% The file keeps the historical colon-separated group shape and column count,
% but column names/content now debug the current metric LevenPhrase algorithm.

lines = readlines('results1.txt');
lines = lines(strlength(lines) > 0);
header = lines(1);
rows = lines(2:end);

% A lightweight parser for the numeric parts. It removes the timestamp and
% colon separators, then extracts numbers. This keeps the example independent
% of your original visualiser.
M = [];
for i = 1:numel(rows)
    txt = rows(i);
    % Drop timestamp: YYYY-MM-DD HH-MM-SS :
    txt = regexprep(txt, '^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2} : ', '');
    nums = regexp(txt, '[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', 'match');
    M(i, 1:numel(nums)) = str2double(nums); %#ok<SAGROW>
end

% Approximate positions after timestamp removal when M.R.A. is non-numeric:
% 1 kind, 2 seq, 3 words, 4 elements, 5 profile_d_delta,
% 6 relative_g3, 7 quality_g3, 8 Lev, 9 D-Lev, 10 Ham,
% 11 Jaro, 12 Jaro-Winkler, then lambda_g1..lambda_g5.
seq = M(:, 2);
g1 = M(:, 13);
g3 = M(:, 15);
g5 = M(:, 17);

figure;
plot(seq, [g1 g3 g5], 'o-');
grid on;
xlabel('sequence number');
ylabel('raw metric LevenPhrase');
legend('gamma=-1', 'gamma=0', 'gamma=1');
title('Metric LevenPhrase results1.txt preview');
