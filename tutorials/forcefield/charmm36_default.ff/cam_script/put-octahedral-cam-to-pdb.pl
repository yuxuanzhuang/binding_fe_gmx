#!/usr/bin/perl -w

$b = $ARGV[0];
shift @ARGV;

while (<>) {
    if (/^ATOM.*CAL CAL/) {
        ($x, $y, $z) = (split)[5..7];
        s/^(ATOM.*?)CAL CAL /$1 D0  CAM/;
        substr($_, 76, 2) = "  ";       # remove element item, othersize residue not match error
        print;

        for $i (1..6) {
            $line = $_;
            substr($line, 14, 1) = $i;
            # pdb use angstrom!
            if ($i == 1) {
                ($xs, $ys, $zs) = map { $_*$b } (1, 0, 0);
            } elsif ($i == 2) {
                ($xs, $ys, $zs) = map { $_*$b } (-1, 0, 0);
            } elsif ($i == 3) {
                ($xs, $ys, $zs) = map { $_*$b } (0, 1, 0);
            } elsif ($i == 4) {
                ($xs, $ys, $zs) = map { $_*$b } (0, -1, 0);
            } elsif ($i == 5) {
                ($xs, $ys, $zs) = map { $_*$b } (0, 0, 1);
            } elsif ($i == 6) {
                ($xs, $ys, $zs) = map { $_*$b } (0, 0, -1);
            }
            substr($line, 30, 8) = sprintf "%8.3f", $x + $xs;
            substr($line, 38, 8) = sprintf "%8.3f", $y + $ys;
            substr($line, 46, 8) = sprintf "%8.3f", $z + $zs;
            print $line;
        }
        next;
    }
    print;
}
